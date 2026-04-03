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
        f"[Qwen-Triton fallback] Matmul Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_matmul(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def _torch_matmul_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    needs_bias_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    grad_x = grad_output @ weight
    grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).T @ x.reshape(-1, x.shape[-1])
    grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0) if needs_bias_grad else None
    return grad_x, grad_weight, grad_bias


if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_n_blocks = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_n_blocks
        pid_n = pid % num_n_blocks

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=mask)

    @triton.jit
    def _bias_add_kernel(
        c_ptr, bias_ptr, M, N,
        stride_cm, stride_cn,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        for col_start in tl.range(0, N, BLOCK_SIZE):
            offs = col_start + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            c = tl.load(c_ptr + row * stride_cm + offs * stride_cn, mask=mask)
            b = tl.load(bias_ptr + offs, mask=mask)
            tl.store(c_ptr + row * stride_cm + offs * stride_cn, c + b, mask=mask)

    @triton.jit
    def _row_sum_kernel(
        grad_out_ptr, grad_bias_ptr,
        M, N,
        stride_gom,
        BLOCK_SIZE: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for row in range(M):
            g = tl.load(grad_out_ptr + row * stride_gom + offs, mask=mask, other=0.0).to(tl.float32)
            acc += g
        tl.store(grad_bias_ptr + offs, acc, mask=mask)


def _triton_matmul_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    original_shape = x.shape
    x_2d = x.contiguous().view(-1, original_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    # weight is (N, K) — we compute x_2d @ weight.T = (M, K) @ (K, N) = (M, N)
    w_t = weight.T.contiguous()  # (K, N)
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        x_2d, w_t, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )

    if bias is not None:
        _bias_add_kernel[(M,)](out, bias, M, N, out.stride(0), out.stride(1), BLOCK_SIZE=min(1024, triton.next_power_of_2(N)))

    return out.view(*original_shape[:-1], N)


def _triton_matmul_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    needs_bias_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    original_shape = x.shape
    x_2d = x.contiguous().view(-1, original_shape[-1])
    grad_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]

    # grad_x = grad_output @ weight: (M, N) @ (N, K) = (M, K)
    grad_x = torch.empty_like(x_2d)
    w_cont = weight.contiguous()  # (N, K)
    grid_gx = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_N"]),)
    _matmul_kernel[grid_gx](
        grad_2d, w_cont, grad_x,
        M, K, N,
        grad_2d.stride(0), grad_2d.stride(1),
        w_cont.stride(0), w_cont.stride(1),
        grad_x.stride(0), grad_x.stride(1),
    )

    # grad_weight = grad_output.T @ x: (N, M) @ (M, K) = (N, K)
    grad_weight = torch.empty_like(weight)
    grad_2d_t = grad_2d.T.contiguous()  # (N, M)
    grid_gw = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_N"]),)
    _matmul_kernel[grid_gw](
        grad_2d_t, x_2d, grad_weight,
        N, K, M,
        grad_2d_t.stride(0), grad_2d_t.stride(1),
        x_2d.stride(0), x_2d.stride(1),
        grad_weight.stride(0), grad_weight.stride(1),
    )

    grad_bias = None
    if needs_bias_grad:
        grad_bias = torch.empty(N, device=grad_output.device, dtype=grad_output.dtype)
        block_size = min(1024, triton.next_power_of_2(N))
        _row_sum_kernel[(triton.cdiv(N, block_size),)](grad_2d, grad_bias, M, N, grad_2d.stride(0), BLOCK_SIZE=block_size)

    return grad_x.view_as(x), grad_weight, grad_bias


class _MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.needs_bias_grad = bias is not None and bias.requires_grad
        return _triton_matmul_forward(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x, weight = ctx.saved_tensors
        try:
            grad_x, grad_weight, grad_bias = _triton_matmul_backward(
                x, weight, grad_output, ctx.needs_bias_grad,
            )
            return grad_x, grad_weight, grad_bias
        except Exception as exc:
            global _TRITON_RUNTIME_OK
            _TRITON_RUNTIME_OK = False
            if os.environ.get("QWEN_TRITON_STRICT") == "1":
                raise RuntimeError("Matmul Triton backward kernel failed and strict Triton mode is enabled.") from exc
            _warn_fallback_once(exc)
            grad_x, grad_weight, grad_bias = _torch_matmul_backward(
                x, weight, grad_output, ctx.needs_bias_grad,
            )
            return grad_x, grad_weight, grad_bias


def triton_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_triton: bool | None = None,
) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (x.is_cuda and weight.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_matmul(x, weight, bias)

    try:
        if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)):
            return _MatmulFunction.apply(x, weight, bias)
        return _triton_matmul_forward(x, weight, bias)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("Matmul Triton kernel failed and strict Triton mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_matmul(x, weight, bias)
