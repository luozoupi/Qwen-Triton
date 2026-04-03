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
        f"[Qwen-Triton fallback] CrossEntropy Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)


if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        ],
        key=["V"],
    )
    @triton.jit
    def _cross_entropy_fwd_kernel(
        logits_ptr, targets_ptr, loss_ptr, lse_ptr,
        N, V,
        stride_lm,
        ignore_index,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= N:
            return

        target = tl.load(targets_ptr + row)
        if target == ignore_index:
            tl.store(loss_ptr + row, 0.0)
            tl.store(lse_ptr + row, 0.0)
            return

        row_ptr = logits_ptr + row * stride_lm

        # Pass 1: find max
        m = tl.full((), float("-inf"), dtype=tl.float32)
        for block_start in tl.range(0, V, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < V
            x = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
            m = tl.maximum(m, tl.max(x, axis=0))

        # Pass 2: sum of exp
        s = tl.zeros((), dtype=tl.float32)
        for block_start in tl.range(0, V, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < V
            x = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
            s += tl.sum(tl.exp(x - m), axis=0)

        log_sum_exp = m + tl.log(s)
        target_logit = tl.load(row_ptr + target).to(tl.float32)
        loss = log_sum_exp - target_logit

        tl.store(loss_ptr + row, loss)
        tl.store(lse_ptr + row, log_sum_exp)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        ],
        key=["V"],
    )
    @triton.jit
    def _cross_entropy_bwd_kernel(
        logits_ptr, targets_ptr, grad_logits_ptr, lse_ptr,
        grad_loss_scalar,
        N, V,
        stride_lm, stride_glm,
        ignore_index,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= N:
            return

        target = tl.load(targets_ptr + row)
        if target == ignore_index:
            for block_start in tl.range(0, V, BLOCK_SIZE):
                offs = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offs < V
                tl.store(grad_logits_ptr + row * stride_glm + offs, 0.0, mask=mask)
            return

        log_sum_exp = tl.load(lse_ptr + row)
        row_logits = logits_ptr + row * stride_lm
        row_grad = grad_logits_ptr + row * stride_glm

        for block_start in tl.range(0, V, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < V
            x = tl.load(row_logits + offs, mask=mask, other=0.0).to(tl.float32)
            softmax_val = tl.exp(x - log_sum_exp)
            is_target = (offs == target)
            g = (softmax_val - tl.where(is_target, 1.0, 0.0)) * grad_loss_scalar
            tl.store(row_grad + offs, g, mask=mask)


def _triton_cross_entropy_forward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, V = logits.shape
    loss_per_token = torch.empty(N, device=logits.device, dtype=torch.float32)
    lse = torch.empty(N, device=logits.device, dtype=torch.float32)

    _cross_entropy_fwd_kernel[(N,)](
        logits, targets, loss_per_token, lse,
        N, V,
        logits.stride(0),
        ignore_index,
    )

    valid_mask = targets != ignore_index
    n_valid = valid_mask.sum()
    loss = loss_per_token.sum() / n_valid.clamp(min=1)
    return loss, lse


class _CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        loss, lse = _triton_cross_entropy_forward(logits, targets, ignore_index)
        ctx.save_for_backward(logits, targets, lse)
        ctx.ignore_index = ignore_index
        valid_mask = targets != ignore_index
        ctx.n_valid = valid_mask.sum().clamp(min=1).item()
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        logits, targets, lse = ctx.saved_tensors
        N, V = logits.shape
        try:
            grad_logits = torch.empty_like(logits)
            grad_loss_scalar = grad_output.item() / ctx.n_valid
            _cross_entropy_bwd_kernel[(N,)](
                logits, targets, grad_logits, lse,
                grad_loss_scalar,
                N, V,
                logits.stride(0), grad_logits.stride(0),
                ctx.ignore_index,
            )
            return grad_logits, None, None
        except Exception as exc:
            global _TRITON_RUNTIME_OK
            _TRITON_RUNTIME_OK = False
            if os.environ.get("QWEN_TRITON_STRICT") == "1":
                raise RuntimeError("CrossEntropy Triton backward failed and strict mode is enabled.") from exc
            _warn_fallback_once(exc)
            loss_ref = _torch_cross_entropy(logits, targets, ctx.ignore_index)
            loss_ref.backward(grad_output)
            return logits.grad, None, None


def triton_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    use_triton: bool | None = None,
) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (logits.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_cross_entropy(logits, targets, ignore_index)

    try:
        if torch.is_grad_enabled() and logits.requires_grad:
            return _CrossEntropyFunction.apply(logits, targets, ignore_index)
        loss, _ = _triton_cross_entropy_forward(logits, targets, ignore_index)
        return loss
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("CrossEntropy Triton kernel failed and strict mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_cross_entropy(logits, targets, ignore_index)
