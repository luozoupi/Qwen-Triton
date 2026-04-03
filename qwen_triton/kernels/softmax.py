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
        f"[Qwen-Triton fallback] Softmax Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _torch_softmax_backward(output: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    s = (grad_output * output).sum(dim=-1, keepdim=True)
    return output * (grad_output - s)


if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_cols"],
    )
    @triton.jit
    def _softmax_kernel(
        x_ptr, out_ptr,
        stride_xm, stride_om,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_x = x_ptr + row * stride_xm
        row_o = out_ptr + row * stride_om

        # Pass 1: find max
        m = tl.full((), float("-inf"), dtype=tl.float32)
        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols
            x = tl.load(row_x + offs, mask=mask, other=float("-inf")).to(tl.float32)
            m = tl.maximum(m, tl.max(x, axis=0))

        # Pass 2: compute sum of exp
        s = tl.zeros((), dtype=tl.float32)
        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols
            x = tl.load(row_x + offs, mask=mask, other=float("-inf")).to(tl.float32)
            s += tl.sum(tl.exp(x - m), axis=0)

        # Pass 3: normalize and store
        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols
            x = tl.load(row_x + offs, mask=mask, other=float("-inf")).to(tl.float32)
            y = tl.exp(x - m) / s
            tl.store(row_o + offs, y, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        ],
        key=["n_cols"],
    )
    @triton.jit
    def _softmax_backward_kernel(
        out_ptr, grad_out_ptr, grad_x_ptr,
        stride_om, stride_gom, stride_gxm,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_o = out_ptr + row * stride_om
        row_go = grad_out_ptr + row * stride_gom
        row_gx = grad_x_ptr + row * stride_gxm

        # Compute dot = sum(grad_out * out)
        dot = tl.zeros((), dtype=tl.float32)
        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols
            o = tl.load(row_o + offs, mask=mask, other=0.0).to(tl.float32)
            go = tl.load(row_go + offs, mask=mask, other=0.0).to(tl.float32)
            dot += tl.sum(o * go, axis=0)

        # Compute grad_x = out * (grad_out - dot)
        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_cols
            o = tl.load(row_o + offs, mask=mask, other=0.0).to(tl.float32)
            go = tl.load(row_go + offs, mask=mask, other=0.0).to(tl.float32)
            gx = o * (go - dot)
            tl.store(row_gx + offs, gx, mask=mask)


def _triton_softmax_forward(x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.contiguous().view(-1, original_shape[-1])
    out = torch.empty_like(x_2d)
    _softmax_kernel[(x_2d.shape[0],)](x_2d, out, x_2d.stride(0), out.stride(0), x_2d.shape[1])
    return out.view(*original_shape)


class _SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = _triton_softmax_forward(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (out,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        try:
            original_shape = out.shape
            out_2d = out.contiguous().view(-1, original_shape[-1])
            grad_2d = grad_output.view(-1, original_shape[-1])
            grad_x = torch.empty_like(out_2d)
            _softmax_backward_kernel[(out_2d.shape[0],)](
                out_2d, grad_2d, grad_x,
                out_2d.stride(0), grad_2d.stride(0), grad_x.stride(0),
                out_2d.shape[1],
            )
            return (grad_x.view(*original_shape),)
        except Exception as exc:
            global _TRITON_RUNTIME_OK
            _TRITON_RUNTIME_OK = False
            if os.environ.get("QWEN_TRITON_STRICT") == "1":
                raise RuntimeError("Softmax Triton backward failed and strict mode is enabled.") from exc
            _warn_fallback_once(exc)
            return (_torch_softmax_backward(out, grad_output),)


def triton_softmax(x: torch.Tensor, use_triton: bool | None = None) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (x.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_softmax(x)

    try:
        if torch.is_grad_enabled() and x.requires_grad:
            return _SoftmaxFunction.apply(x)
        return _triton_softmax_forward(x)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("Softmax Triton kernel failed and strict mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_softmax(x)
