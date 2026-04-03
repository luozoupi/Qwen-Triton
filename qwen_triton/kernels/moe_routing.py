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
        f"[Qwen-Triton fallback] MoE routing Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Top-k (small k, per-row)
# ---------------------------------------------------------------------------

def _torch_topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(x, k, dim=-1)


if _TRITON_AVAILABLE:
    @triton.jit
    def _topk_k2_kernel(
        x_ptr, vals_ptr, idxs_ptr,
        N, C,
        stride_xn, stride_vn, stride_in,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Top-2 selection per row. Each program handles one row."""
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_xn
        v1 = tl.full((), float("-inf"), dtype=tl.float32)
        v2 = tl.full((), float("-inf"), dtype=tl.float32)
        i1 = tl.zeros((), dtype=tl.int64)
        i2 = tl.zeros((), dtype=tl.int64)

        for block_start in tl.range(0, C, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < C
            x = tl.load(x_row + offs, mask=mask, other=float("-inf")).to(tl.float32)

            for j in tl.static_range(0, BLOCK_SIZE):
                val = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == j, x, tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32)))
                col = block_start + j
                valid = col < C
                is_new_top1 = valid & (val > v1)
                is_new_top2 = valid & (~is_new_top1) & (val > v2)
                # Update top-2
                new_v2 = tl.where(is_new_top1, v1, tl.where(is_new_top2, val, v2))
                new_i2 = tl.where(is_new_top1, i1, tl.where(is_new_top2, col.to(tl.int64), i2))
                v2 = new_v2
                i2 = new_i2
                v1 = tl.where(is_new_top1, val, v1)
                i1 = tl.where(is_new_top1, col.to(tl.int64), i1)

        tl.store(vals_ptr + row * stride_vn, v1)
        tl.store(vals_ptr + row * stride_vn + 1, v2)
        tl.store(idxs_ptr + row * stride_in, i1)
        tl.store(idxs_ptr + row * stride_in + 1, i2)

    @triton.jit
    def _topk_general_kernel(
        x_ptr, vals_ptr, idxs_ptr,
        N, C, K,
        stride_xn, stride_vn, stride_in,
        BLOCK_SIZE: tl.constexpr,
        MAX_K: tl.constexpr,
    ):
        """General top-k per row. Uses iterative max-finding for small K."""
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_xn
        neg_inf = float("-inf")

        for ki in tl.static_range(0, MAX_K):
            if ki < K:
                best_val = tl.full((), neg_inf, dtype=tl.float32)
                best_idx = tl.zeros((), dtype=tl.int64)
                for block_start in tl.range(0, C, BLOCK_SIZE):
                    offs = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = offs < C
                    x = tl.load(x_row + offs, mask=mask, other=neg_inf).to(tl.float32)
                    local_max = tl.max(x, axis=0)
                    if local_max > best_val:
                        # Find the index of the max in this block
                        is_max = (x == local_max) & mask
                        # Get first True index
                        local_idx = tl.min(tl.where(is_max, offs, C), axis=0)
                        best_val = local_max
                        best_idx = local_idx.to(tl.int64)

                tl.store(vals_ptr + row * stride_vn + ki, best_val)
                tl.store(idxs_ptr + row * stride_in + ki, best_idx)
                # Mark this position as used
                tl.store(x_row + best_idx, neg_inf)


def _triton_topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    N, C = x.shape
    vals = torch.empty((N, k), device=x.device, dtype=x.dtype)
    idxs = torch.empty((N, k), device=x.device, dtype=torch.int64)

    if k == 2 and C <= 256:
        BLOCK_SIZE = triton.next_power_of_2(C)
        _topk_k2_kernel[(N,)](
            x, vals, idxs,
            N, C,
            x.stride(0), vals.stride(0), idxs.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # General case: iterative max-finding (modifies input!)
        x_work = x.clone()
        BLOCK_SIZE = min(1024, triton.next_power_of_2(C))
        MAX_K = max(k, 2)  # constexpr needs compile-time known bound
        _topk_general_kernel[(N,)](
            x_work, vals, idxs,
            N, C, k,
            x_work.stride(0), vals.stride(0), idxs.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_K=MAX_K,
        )
    return vals, idxs


class _TopkFunction(torch.autograd.Function):
    """Topk is not differentiable w.r.t. indices; gradient just gathers back."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        vals, idxs = _triton_topk(x, k)
        ctx.save_for_backward(idxs)
        ctx.input_shape = x.shape
        return vals, idxs

    @staticmethod
    def backward(ctx, grad_vals: torch.Tensor, grad_idxs: torch.Tensor) -> tuple[torch.Tensor, None]:
        (idxs,) = ctx.saved_tensors
        grad_x = torch.zeros(ctx.input_shape, device=grad_vals.device, dtype=grad_vals.dtype)
        grad_x.scatter_add_(1, idxs, grad_vals)
        return grad_x, None


# ---------------------------------------------------------------------------
# One-hot (Triton)
# ---------------------------------------------------------------------------

def _torch_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices, num_classes)


if _TRITON_AVAILABLE:
    @triton.jit
    def _one_hot_kernel(
        idx_ptr, out_ptr,
        n_elements, num_classes,
        stride_in, stride_on, stride_oc,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        idx = tl.load(idx_ptr + row * stride_in)
        offs_c = tl.arange(0, BLOCK_C)
        mask = offs_c < num_classes
        vals = tl.where(offs_c == idx, 1, 0)
        tl.store(out_ptr + row * stride_on + offs_c * stride_oc, vals, mask=mask)


def _triton_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    flat = indices.contiguous().view(-1)
    N = flat.numel()
    out = torch.zeros((N, num_classes), device=indices.device, dtype=torch.int64)
    BLOCK_C = triton.next_power_of_2(num_classes)
    _one_hot_kernel[(N,)](
        flat, out,
        N, num_classes,
        flat.stride(0), out.stride(0), out.stride(1),
        BLOCK_C=BLOCK_C,
    )
    return out.view(*indices.shape, num_classes)


# ---------------------------------------------------------------------------
# Index-add (weighted scatter-add)
# ---------------------------------------------------------------------------

def _torch_index_add(
    target: torch.Tensor,
    source: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    target.index_add_(0, index, source)
    return target


if _TRITON_AVAILABLE:
    @triton.jit
    def _index_add_kernel(
        target_ptr, source_ptr, index_ptr,
        n_source, D,
        stride_tm, stride_td,
        stride_sm, stride_sd,
        BLOCK_D: tl.constexpr,
    ):
        src_row = tl.program_id(0)
        if src_row >= n_source:
            return
        tgt_row = tl.load(index_ptr + src_row)
        for d_start in tl.range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask = offs_d < D
            val = tl.load(source_ptr + src_row * stride_sm + offs_d * stride_sd, mask=mask, other=0.0)
            tl.atomic_add(target_ptr + tgt_row * stride_tm + offs_d * stride_td, val, mask=mask)


def _triton_index_add(
    target: torch.Tensor,
    source: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    n_source = source.shape[0]
    D = source.shape[1]
    BLOCK_D = min(1024, triton.next_power_of_2(D))
    _index_add_kernel[(n_source,)](
        target, source, index,
        n_source, D,
        target.stride(0), target.stride(1),
        source.stride(0), source.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return target


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------

def triton_topk(
    x: torch.Tensor,
    k: int,
    use_triton: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _TRITON_RUNTIME_OK
    use_triton = (x.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_topk(x, k)

    try:
        if torch.is_grad_enabled() and x.requires_grad:
            return _TopkFunction.apply(x, k)
        return _triton_topk(x, k)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("TopK Triton kernel failed and strict mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_topk(x, k)


def triton_one_hot(
    indices: torch.Tensor,
    num_classes: int,
    use_triton: bool | None = None,
) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (indices.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_one_hot(indices, num_classes)

    try:
        return _triton_one_hot(indices, num_classes)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("OneHot Triton kernel failed and strict mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_one_hot(indices, num_classes)


class _IndexAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target: torch.Tensor, source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.source_shape = source.shape
        ctx.target_shape = target.shape
        # We must not modify target in-place when it's part of the autograd graph;
        # create a new output tensor instead.
        out = target.clone()
        _triton_index_add(out, source, index)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        (index,) = ctx.saved_tensors
        # grad_target = grad_output (identity)
        grad_target = grad_output
        # grad_source = grad_output[index] (gather)
        grad_source = grad_output[index]
        return grad_target, grad_source, None


def triton_index_add(
    target: torch.Tensor,
    source: torch.Tensor,
    index: torch.Tensor,
    use_triton: bool | None = None,
) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (target.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_index_add(target, source, index)

    try:
        if torch.is_grad_enabled() and (target.requires_grad or source.requires_grad):
            return _IndexAddFunction.apply(target, source, index)
        _triton_index_add(target, source, index)
        return target
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("IndexAdd Triton kernel failed and strict mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_index_add(target, source, index)
