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
        f"[Qwen-Triton fallback] FlashAttention Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = True,
    num_kv_groups: int = 1,
) -> torch.Tensor:
    if num_kv_groups > 1:
        B, nkv, S, D = k.shape
        k = k[:, :, None, :, :].expand(B, nkv, num_kv_groups, S, D).reshape(B, nkv * num_kv_groups, S, D)
        v = v[:, :, None, :, :].expand(B, nkv, num_kv_groups, S, D).reshape(B, nkv * num_kv_groups, S, D)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=scale,
    )


if _TRITON_AVAILABLE:
    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out, LSE,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_lb, stride_lh,
        N_CTX_Q, N_CTX_K,
        scale,
        num_kv_groups,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        # Decompose pid_bh into batch and head
        num_q_heads = tl.num_programs(1) // (stride_qb // stride_qh) if stride_qh > 0 else 1
        # Actually, we pass B*num_q_heads as grid dim 1
        # pid_bh encodes batch * num_q_heads + q_head
        # For GQA, map q_head to kv_head
        pid_b = pid_bh // (stride_qb // stride_qh) if stride_qh > 0 else pid_bh
        pid_h = pid_bh % (stride_qb // stride_qh) if stride_qh > 0 else 0
        pid_kv_h = pid_h // num_kv_groups

        # Offsets for Q block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        # Q pointers
        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < BLOCK_D)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        # Initialize accumulators
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # K, V base pointers (using kv head)
        k_base = K + pid_b * stride_kb + pid_kv_h * stride_kh
        v_base = V + pid_b * stride_vb + pid_kv_h * stride_vh

        # Determine K/V iteration range
        if IS_CAUSAL:
            kv_end = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX_K)
        else:
            kv_end = N_CTX_K

        for kv_start in range(0, kv_end, BLOCK_N):
            offs_n = kv_start + tl.arange(0, BLOCK_N)

            # Load K block
            k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
            k_mask = (offs_n[None, :] < N_CTX_K) & (offs_d[:, None] < BLOCK_D)
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Compute QK^T
            qk = tl.dot(q, k) * scale  # (BLOCK_M, BLOCK_N)

            # Apply causal mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Mask out-of-bounds
            qk = tl.where(offs_n[None, :] < N_CTX_K, qk, float("-inf"))

            # Online softmax update
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_i = l_i * alpha + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
            acc = acc * alpha[:, None]

            # Load V block and accumulate
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v_mask = (offs_n[:, None] < N_CTX_K) & (offs_d[None, :] < BLOCK_D)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
            p = tl.exp(qk - m_new[:, None])
            acc += tl.dot(p.to(v.dtype), v)

            m_i = m_new

        # Final normalization
        acc = acc / l_i[:, None]
        lse = m_i + tl.log(l_i)

        # Store output
        o_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        o_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < BLOCK_D)
        tl.store(o_ptrs, acc, mask=o_mask)

        # Store LSE
        lse_ptrs = LSE + pid_b * stride_lb + pid_h * stride_lh + offs_m
        lse_mask = offs_m < N_CTX_Q
        tl.store(lse_ptrs, lse, mask=lse_mask)

    @triton.jit
    def _flash_attn_bwd_preprocess(
        Out, dOut, Delta,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_db, stride_dh,
        N_CTX, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        num_heads = tl.num_programs(1) // (stride_ob // stride_oh) if stride_oh > 0 else 1
        pid_b = pid_bh // (stride_ob // stride_oh) if stride_oh > 0 else pid_bh
        pid_h = pid_bh % (stride_ob // stride_oh) if stride_oh > 0 else 0

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < BLOCK_D)

        o = tl.load(Out + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(dOut + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, mask=mask, other=0.0).to(tl.float32)

        delta = tl.sum(o * do, axis=1)
        delta_ptrs = Delta + pid_b * stride_db + pid_h * stride_dh + offs_m
        tl.store(delta_ptrs, delta, mask=offs_m < N_CTX)

    @triton.jit
    def _flash_attn_bwd_kernel(
        Q, K, V, Out, dOut,
        dQ, dK, dV,
        LSE, Delta,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_lb, stride_lh,
        stride_db, stride_dh,
        N_CTX_Q, N_CTX_K,
        scale,
        num_kv_groups,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        # For backward, we iterate over K/V blocks (pid_n) and accumulate dK, dV
        # We need to handle GQA: multiple Q heads map to the same K/V head
        # pid_bh here encodes batch * num_kv_heads + kv_head
        num_kv_heads = stride_kb // stride_kh if stride_kh > 0 else 1
        pid_b = pid_bh // num_kv_heads
        pid_kv_h = pid_bh % num_kv_heads

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        # Load K, V block for this kv_head
        k_ptrs = K + pid_b * stride_kb + pid_kv_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + pid_b * stride_vb + pid_kv_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k_mask = (offs_n[:, None] < N_CTX_K) & (offs_d[None, :] < BLOCK_D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

        # Iterate over all Q heads that map to this KV head
        for g in range(num_kv_groups):
            q_head = pid_kv_h * num_kv_groups + g
            q_base = Q + pid_b * stride_qb + q_head * stride_qh
            do_base = dOut + pid_b * stride_ob + q_head * stride_oh
            dq_base = dQ + pid_b * stride_qb + q_head * stride_qh
            lse_base = LSE + pid_b * stride_lb + q_head * stride_lh
            delta_base = Delta + pid_b * stride_db + q_head * stride_dh

            # Determine Q range to iterate
            if IS_CAUSAL:
                q_start = pid_n * BLOCK_N
            else:
                q_start = 0

            for q_block_start in range(q_start, N_CTX_Q, BLOCK_M):
                offs_m = q_block_start + tl.arange(0, BLOCK_M)

                # Load Q, dO, LSE, Delta for this Q block
                q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
                q_mask = (offs_m[:, None] < N_CTX_Q) & (offs_d[None, :] < BLOCK_D)
                q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
                do = tl.load(do_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, mask=q_mask, other=0.0).to(tl.float32)
                lse = tl.load(lse_base + offs_m, mask=offs_m < N_CTX_Q, other=0.0)
                delta = tl.load(delta_base + offs_m, mask=offs_m < N_CTX_Q, other=0.0)

                # Recompute attention: S = Q @ K^T * scale
                s = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_M, BLOCK_N)

                if IS_CAUSAL:
                    causal_mask = offs_m[:, None] >= offs_n[None, :]
                    s = tl.where(causal_mask, s, float("-inf"))
                s = tl.where(offs_n[None, :] < N_CTX_K, s, float("-inf"))

                # P = exp(S - LSE)
                p = tl.exp(s - lse[:, None])

                # dV += P^T @ dO
                dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)

                # dP = dO @ V^T
                dp = tl.dot(do, tl.trans(v))

                # dS = P * (dP - Delta)
                ds = p * (dp - delta[:, None]) * scale

                # dK += dS^T @ Q
                dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)

                # dQ += dS @ K (atomic add since multiple blocks contribute)
                dq_update = tl.dot(ds.to(k.dtype), k)
                dq_ptrs = dq_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
                tl.atomic_add(dq_ptrs, dq_update, mask=q_mask)

        # Store dK, dV
        dk_ptrs = dK + pid_b * stride_kb + pid_kv_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        dv_ptrs = dV + pid_b * stride_vb + pid_kv_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        tl.store(dk_ptrs, dk_acc, mask=k_mask)
        tl.store(dv_ptrs, dv_acc, mask=k_mask)


def _triton_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
    num_kv_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, nq, S_q, D = q.shape
    _, nkv, S_k, _ = k.shape
    assert D <= 128, f"Head dim {D} > 128 not supported"
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_M = 32
    BLOCK_N = 32

    out = torch.empty_like(q)
    lse = torch.empty((B, nq, S_q), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(S_q, BLOCK_M), B * nq)
    _flash_attn_fwd_kernel[grid](
        q, k, v, out, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1),
        S_q, S_k,
        scale,
        num_kv_groups,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out, lse


def _triton_flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    is_causal: bool,
    num_kv_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, nq, S_q, D = q.shape
    _, nkv, S_k, _ = k.shape
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_M = 32
    BLOCK_N = 32

    # Precompute Delta = rowsum(O * dO)
    delta = torch.empty((B, nq, S_q), device=q.device, dtype=torch.float32)
    grid_pre = (triton.cdiv(S_q, BLOCK_M), B * nq)
    _flash_attn_bwd_preprocess[grid_pre](
        out, grad_output, delta,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        delta.stride(0), delta.stride(1),
        S_q, BLOCK_D=BLOCK_D, BLOCK_M=BLOCK_M,
    )

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    grid_bwd = (triton.cdiv(S_k, BLOCK_N), B * nkv)
    _flash_attn_bwd_kernel[grid_bwd](
        q, k, v, out, grad_output,
        dq, dk, dv,
        lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        S_q, S_k,
        scale,
        num_kv_groups,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return dq, dk, dv


class _FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        is_causal: bool,
        num_kv_groups: int,
    ) -> torch.Tensor:
        out, lse = _triton_flash_attention_forward(q, k, v, scale, is_causal, num_kv_groups)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.num_kv_groups = num_kv_groups
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q, k, v, out, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        try:
            dq, dk, dv = _triton_flash_attention_backward(
                q, k, v, out, grad_output, lse,
                ctx.scale, ctx.is_causal, ctx.num_kv_groups,
            )
            return dq, dk, dv, None, None, None
        except Exception as exc:
            global _TRITON_RUNTIME_OK
            _TRITON_RUNTIME_OK = False
            if os.environ.get("QWEN_TRITON_STRICT") == "1":
                raise RuntimeError("FlashAttention Triton backward failed and strict mode is enabled.") from exc
            _warn_fallback_once(exc)
            out_ref = _torch_flash_attention(q, k, v, ctx.scale, ctx.is_causal, ctx.num_kv_groups)
            out_ref.backward(grad_output)
            return q.grad, k.grad, v.grad, None, None, None


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = True,
    num_kv_groups: int = 1,
    use_triton: bool | None = None,
) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (q.is_cuda and k.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_flash_attention(q, k, v, scale, is_causal, num_kv_groups)

    try:
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
            return _FlashAttentionFunction.apply(q, k, v, scale, is_causal, num_kv_groups)
        out, _lse = _triton_flash_attention_forward(q, k, v, scale, is_causal, num_kv_groups)
        return out
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("FlashAttention Triton kernel failed and strict mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_flash_attention(q, k, v, scale, is_causal, num_kv_groups)
