from __future__ import annotations

import os
import warnings

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_TRITON_RUNTIME_OK = _TRITON_AVAILABLE
_FALLBACK_WARNED = False


def _warn_fallback_once(exc: Exception) -> None:
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    warnings.warn(
        f"[Qwen-Triton fallback] ResidualAdd Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_residual_add(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return x + residual


if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _residual_add_kernel(
        x_ptr, residual_ptr, out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        r = tl.load(residual_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + r, mask=mask)


def _triton_residual_add_forward(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _residual_add_kernel[grid](
        x.contiguous().view(-1),
        residual.contiguous().view(-1),
        out.view(-1),
        n,
    )
    return out


class _ResidualAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return _triton_residual_add_forward(x, residual)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return grad_output, grad_output


def residual_add(x: torch.Tensor, residual: torch.Tensor, use_triton: bool | None = None) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (x.is_cuda and residual.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_residual_add(x, residual)

    try:
        if torch.is_grad_enabled() and (x.requires_grad or residual.requires_grad):
            return _ResidualAddFunction.apply(x, residual)
        return _triton_residual_add_forward(x, residual)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("ResidualAdd Triton kernel failed and strict mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_residual_add(x, residual)
