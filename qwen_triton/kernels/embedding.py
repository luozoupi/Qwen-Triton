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
        f"[Qwen-Triton fallback] Embedding Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_embedding(input_ids: torch.Tensor, weight: torch.Tensor, padding_idx: int | None = None) -> torch.Tensor:
    return torch.nn.functional.embedding(input_ids, weight, padding_idx=padding_idx)


if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        ],
        key=["embed_dim"],
    )
    @triton.jit
    def _embedding_kernel(
        ids_ptr, weight_ptr, out_ptr,
        n_tokens, embed_dim,
        stride_wv, stride_we,
        stride_om, stride_oe,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        if token_idx >= n_tokens:
            return
        token_id = tl.load(ids_ptr + token_idx)

        for block_start in tl.range(0, embed_dim, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < embed_dim
            w = tl.load(weight_ptr + token_id * stride_wv + offs * stride_we, mask=mask)
            tl.store(out_ptr + token_idx * stride_om + offs * stride_oe, w, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        ],
        key=["embed_dim"],
    )
    @triton.jit
    def _embedding_backward_kernel(
        ids_ptr, grad_out_ptr, grad_weight_ptr,
        n_tokens, embed_dim,
        stride_gom, stride_goe,
        stride_gwv, stride_gwe,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        if token_idx >= n_tokens:
            return
        token_id = tl.load(ids_ptr + token_idx)

        for block_start in tl.range(0, embed_dim, BLOCK_SIZE):
            offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < embed_dim
            g = tl.load(grad_out_ptr + token_idx * stride_gom + offs * stride_goe, mask=mask, other=0.0)
            tl.atomic_add(grad_weight_ptr + token_id * stride_gwv + offs * stride_gwe, g, mask=mask)


def _triton_embedding_forward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    original_shape = input_ids.shape
    ids_flat = input_ids.contiguous().view(-1)
    n_tokens = ids_flat.numel()
    embed_dim = weight.shape[1]
    out = torch.empty((n_tokens, embed_dim), device=weight.device, dtype=weight.dtype)
    _embedding_kernel[(n_tokens,)](
        ids_flat, weight, out,
        n_tokens, embed_dim,
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out.view(*original_shape, embed_dim)


class _EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_ids: torch.Tensor, weight: torch.Tensor, padding_idx: int | None) -> torch.Tensor:
        ctx.save_for_backward(input_ids)
        ctx.num_embeddings = weight.shape[0]
        ctx.embed_dim = weight.shape[1]
        ctx.padding_idx = padding_idx
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device
        return _triton_embedding_forward(input_ids, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None]:
        (input_ids,) = ctx.saved_tensors
        try:
            ids_flat = input_ids.contiguous().view(-1)
            grad_flat = grad_output.contiguous().view(-1, ctx.embed_dim)
            n_tokens = ids_flat.numel()
            grad_weight = torch.zeros(
                (ctx.num_embeddings, ctx.embed_dim),
                device=ctx.weight_device,
                dtype=torch.float32,
            )
            _embedding_backward_kernel[(n_tokens,)](
                ids_flat, grad_flat, grad_weight,
                n_tokens, ctx.embed_dim,
                grad_flat.stride(0), grad_flat.stride(1),
                grad_weight.stride(0), grad_weight.stride(1),
            )
            if ctx.padding_idx is not None:
                grad_weight[ctx.padding_idx] = 0
            return None, grad_weight.to(ctx.weight_dtype), None
        except Exception as exc:
            global _TRITON_RUNTIME_OK
            _TRITON_RUNTIME_OK = False
            if os.environ.get("QWEN_TRITON_STRICT") == "1":
                raise RuntimeError("Embedding Triton backward failed and strict mode is enabled.") from exc
            _warn_fallback_once(exc)
            ids_flat = input_ids.view(-1)
            grad_flat = grad_output.view(-1, ctx.embed_dim)
            grad_weight = torch.zeros(
                (ctx.num_embeddings, ctx.embed_dim),
                device=ctx.weight_device,
                dtype=ctx.weight_dtype,
            )
            grad_weight.index_add_(0, ids_flat, grad_flat.to(ctx.weight_dtype))
            if ctx.padding_idx is not None:
                grad_weight[ctx.padding_idx] = 0
            return None, grad_weight, None


def triton_embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int | None = None,
    use_triton: bool | None = None,
) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (weight.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_embedding(input_ids, weight, padding_idx)

    try:
        if torch.is_grad_enabled() and weight.requires_grad:
            return _EmbeddingFunction.apply(input_ids, weight, padding_idx)
        return _triton_embedding_forward(input_ids, weight)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("Embedding Triton kernel failed and strict mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_embedding(input_ids, weight, padding_idx)
