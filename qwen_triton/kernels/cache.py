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
        f"[Qwen-Triton fallback] KV cache Triton helper unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


if _TRITON_AVAILABLE:
    @triton.jit
    def _append_kv_kernel(
        prefix_ptr,
        suffix_ptr,
        out_ptr,
        batch_size,
        num_heads,
        old_seq,
        new_seq,
        head_dim,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        d = offsets % head_dim
        tmp = offsets // head_dim
        total_seq = old_seq + new_seq
        s = tmp % total_seq
        tmp = tmp // total_seq
        h = tmp % num_heads
        b = tmp // num_heads

        prefix_index = (((b * num_heads + h) * old_seq + s) * head_dim) + d
        suffix_index = (((b * num_heads + h) * new_seq + (s - old_seq)) * head_dim) + d
        out_index = offsets
        is_prefix = s < old_seq

        prefix_val = tl.load(prefix_ptr + prefix_index, mask=mask & is_prefix, other=0.0)
        suffix_val = tl.load(suffix_ptr + suffix_index, mask=mask & (~is_prefix), other=0.0)
        out = tl.where(is_prefix, prefix_val, suffix_val)
        tl.store(out_ptr + out_index, out, mask=mask)

    @triton.jit
    def _write_kv_kernel(
        cache_ptr,
        values_ptr,
        positions_ptr,
        num_heads,
        seq_len,
        head_dim,
        cache_seq,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        d = offsets % head_dim
        tmp = offsets // head_dim
        s = tmp % seq_len
        tmp = tmp // seq_len
        h = tmp % num_heads
        b = tmp // num_heads
        pos = tl.load(positions_ptr + s, mask=mask, other=0).to(tl.int64)

        src_index = (((b * num_heads + h) * seq_len + s) * head_dim) + d
        dst_index = (((b * num_heads + h) * cache_seq + pos) * head_dim) + d
        value = tl.load(values_ptr + src_index, mask=mask, other=0.0)
        tl.store(cache_ptr + dst_index, value, mask=mask)


def append_attention_kv(prefix: torch.Tensor | None, suffix: torch.Tensor) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    if prefix is None:
        return suffix
    if not (prefix.is_cuda and suffix.is_cuda and _TRITON_AVAILABLE and _TRITON_RUNTIME_OK):
        return torch.cat((prefix, suffix), dim=-2)

    if prefix.dim() != 4 or suffix.dim() != 4:
        return torch.cat((prefix, suffix), dim=-2)

    batch_size, num_heads, old_seq, head_dim = prefix.shape
    new_seq = suffix.shape[-2]
    out = torch.empty(
        (batch_size, num_heads, old_seq + new_seq, head_dim),
        device=prefix.device,
        dtype=prefix.dtype,
    )
    total_elements = out.numel()
    try:
        grid = (triton.cdiv(total_elements, 256),)
        _append_kv_kernel[grid](
            prefix.contiguous(),
            suffix.contiguous(),
            out,
            batch_size,
            num_heads,
            old_seq,
            new_seq,
            head_dim,
            total_elements,
            BLOCK_SIZE=256,
        )
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("KV cache Triton helper failed and strict Triton mode is enabled.") from exc
        _warn_fallback_once(exc)
        return torch.cat((prefix, suffix), dim=-2)
    return out


def write_attention_kv(
    cache: torch.Tensor,
    values: torch.Tensor,
    positions: torch.LongTensor,
) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    if values.numel() == 0:
        return cache

    positions = positions.to(device=cache.device, dtype=torch.long).contiguous()
    if not (cache.is_cuda and values.is_cuda and positions.is_cuda and _TRITON_AVAILABLE and _TRITON_RUNTIME_OK):
        cache[:, :, positions, :] = values
        return cache

    if cache.dim() != 4 or values.dim() != 4:
        cache[:, :, positions, :] = values
        return cache

    batch_size, num_heads, seq_len, head_dim = values.shape
    total_elements = values.numel()
    try:
        grid = (triton.cdiv(total_elements, 256),)
        _write_kv_kernel[grid](
            cache,
            values.contiguous(),
            positions,
            num_heads,
            seq_len,
            head_dim,
            cache.shape[-2],
            total_elements,
            BLOCK_SIZE=256,
        )
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("KV cache Triton write helper failed and strict Triton mode is enabled.") from exc
        _warn_fallback_once(exc)
        cache[:, :, positions, :] = values
    return cache
