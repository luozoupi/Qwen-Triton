import pytest
import torch

from qwen_triton.kernels import apply_rope, gated_delta_rule_sequence, rmsnorm, sigmoid_mul, silu_mul, write_attention_kv
from qwen_triton.ops import apply_rope_cuda_op, get_rope_cuda_op_error, load_rope_cuda_op


def _torch_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    half = rotary_dim // 2
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q1, q2 = q_rot[..., :half], q_rot[..., half:]
    k1, k2 = k_rot[..., :half], k_rot[..., half:]
    q_rot = torch.cat((q1 * cos[..., :half] - q2 * sin[..., :half], q2 * cos[..., :half] + q1 * sin[..., :half]), dim=-1)
    k_rot = torch.cat((k1 * cos[..., :half] - k2 * sin[..., :half], k2 * cos[..., :half] + k1 * sin[..., :half]), dim=-1)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_rmsnorm_kernel_matches_torch() -> None:
    x = torch.randn(8, 1024, device="cuda", dtype=torch.float32)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32)
    expected = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)) * weight
    actual = rmsnorm(x, weight, eps=1e-6, one_plus_weight=False, use_triton=True)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_rmsnorm_kernel_backward_matches_torch() -> None:
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(256, device="cuda", dtype=torch.float32, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x)

    actual = rmsnorm(x, weight, eps=1e-6, one_plus_weight=False, use_triton=True)
    expected = rmsnorm(ref_x, ref_weight, eps=1e-6, one_plus_weight=False, use_triton=False)
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_silu_mul_kernel_matches_torch() -> None:
    gate = torch.randn(8, 3072, device="cuda", dtype=torch.float32)
    up = torch.randn(8, 3072, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(silu_mul(gate, up, use_triton=True), torch.nn.functional.silu(gate) * up, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_silu_mul_kernel_backward_matches_torch() -> None:
    gate = torch.randn(4, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    up = torch.randn(4, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    ref_gate = gate.detach().clone().requires_grad_(True)
    ref_up = up.detach().clone().requires_grad_(True)
    grad = torch.randn_like(gate)

    actual = silu_mul(gate, up, use_triton=True)
    expected = silu_mul(ref_gate, ref_up, use_triton=False)
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(gate.grad, ref_gate.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(up.grad, ref_up.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_sigmoid_mul_kernel_matches_torch() -> None:
    x = torch.randn(8, 3072, device="cuda", dtype=torch.float32)
    gate = torch.randn(8, 3072, device="cuda", dtype=torch.float32)
    expected = x * torch.sigmoid(gate)
    actual = sigmoid_mul(x, gate, use_triton=True)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_sigmoid_mul_kernel_backward_matches_torch() -> None:
    x = torch.randn(4, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    gate = torch.randn(4, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    ref_gate = gate.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x)

    actual = sigmoid_mul(x, gate, use_triton=True)
    expected = sigmoid_mul(ref_x, ref_gate, use_triton=False)
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gate.grad, ref_gate.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_rope_kernel_matches_torch() -> None:
    q = torch.randn(2, 4, 5, 16, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 4, 5, 16, device="cuda", dtype=torch.float32)
    cos = torch.randn(2, 5, 8, device="cuda", dtype=torch.float32)
    sin = torch.randn(2, 5, 8, device="cuda", dtype=torch.float32)
    expected_q, expected_k = _torch_rope(q, k, cos, sin)
    actual_q, actual_k = apply_rope(q, k, cos, sin, use_triton=True)
    torch.testing.assert_close(actual_q, expected_q, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_rope_kernel_backward_matches_torch() -> None:
    q = torch.randn(2, 4, 5, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 2, 5, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    cos = torch.randn(2, 5, 8, device="cuda", dtype=torch.float32)
    sin = torch.randn(2, 5, 8, device="cuda", dtype=torch.float32)
    grad_q = torch.randn_like(q)
    grad_k = torch.randn_like(k)

    actual_q, actual_k = apply_rope(q, k, cos, sin, use_triton=True, backend="triton")
    expected_q, expected_k = apply_rope(ref_q, ref_k, cos, sin, use_triton=False, backend="torch")
    (actual_q * grad_q).sum().backward(retain_graph=True)
    (actual_k * grad_k).sum().backward()
    (expected_q * grad_q).sum().backward(retain_graph=True)
    (expected_k * grad_k).sum().backward()

    torch.testing.assert_close(q.grad, ref_q.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k.grad, ref_k.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_rope_kernel_backward_matches_torch_for_gqa_bf16() -> None:
    q = torch.randn(2, 4, 5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 2, 5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_q = q.detach().clone().requires_grad_(True)
    ref_k = k.detach().clone().requires_grad_(True)
    cos = torch.randn(2, 5, 8, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(2, 5, 8, device="cuda", dtype=torch.bfloat16)
    grad_q = torch.randn_like(q)
    grad_k = torch.randn_like(k)

    actual_q, actual_k = apply_rope(q, k, cos, sin, use_triton=True, backend="triton")
    expected_q, expected_k = apply_rope(ref_q, ref_k, cos, sin, use_triton=False, backend="torch")
    (actual_q.float() * grad_q.float()).sum().backward(retain_graph=True)
    (actual_k.float() * grad_k.float()).sum().backward()
    (expected_q.float() * grad_q.float()).sum().backward(retain_graph=True)
    (expected_k.float() * grad_k.float()).sum().backward()

    torch.testing.assert_close(q.grad.float(), ref_q.grad.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad.float(), ref_k.grad.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_rope_kernel_matches_torch_for_gqa_bf16() -> None:
    q = torch.randn(2, 4, 5, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 5, 16, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(2, 5, 8, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(2, 5, 8, device="cuda", dtype=torch.bfloat16)
    expected_q, expected_k = _torch_rope(q, k, cos, sin)
    actual_q, actual_k = apply_rope(q, k, cos, sin, use_triton=True, backend="triton")
    torch.testing.assert_close(actual_q.float(), expected_q.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_k.float(), expected_k.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_rope_cuda_custom_op_matches_torch_for_gqa_bf16() -> None:
    if not load_rope_cuda_op():
        pytest.skip(f"CUDA custom op unavailable: {get_rope_cuda_op_error()}")
    q = torch.randn(2, 4, 5, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 5, 16, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(2, 5, 8, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(2, 5, 8, device="cuda", dtype=torch.bfloat16)
    expected_q, expected_k = _torch_rope(q, k, cos, sin)
    actual_q, actual_k = apply_rope_cuda_op(q, k, cos, sin)
    torch.testing.assert_close(actual_q.float(), expected_q.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_k.float(), expected_k.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_write_attention_kv_matches_torch_update() -> None:
    cache = torch.zeros(2, 3, 16, 8, device="cuda", dtype=torch.float32)
    values = torch.randn(2, 3, 5, 8, device="cuda", dtype=torch.float32)
    positions = torch.tensor([0, 1, 4, 7, 8], device="cuda", dtype=torch.long)
    expected = cache.clone()
    expected[:, :, positions, :] = values
    actual = cache.clone()
    write_attention_kv(actual, values, positions)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests")
def test_gated_delta_rule_matches_torch_fallback() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 4, 3, 8, device="cuda", dtype=torch.float32)
    key = torch.randn(2, 4, 3, 8, device="cuda", dtype=torch.float32)
    value = torch.randn(2, 4, 3, 6, device="cuda", dtype=torch.float32)
    decay = torch.randn(2, 4, 3, device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(2, 4, 3, device="cuda", dtype=torch.float32))
    expected_out, expected_state = gated_delta_rule_sequence(query, key, value, decay, beta, use_triton=False)
    actual_out, actual_state = gated_delta_rule_sequence(query, key, value, decay, beta, use_triton=True)
    torch.testing.assert_close(actual_out, expected_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_state, expected_state, atol=1e-4, rtol=1e-4)
