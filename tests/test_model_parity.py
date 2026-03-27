import torch

from qwen_triton.models import QwenTritonForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM


def _assert_state_loads(our_model: QwenTritonForCausalLM, hf_model: torch.nn.Module) -> None:
    missing, unexpected = our_model.load_state_dict(hf_model.state_dict(), strict=False)
    assert not unexpected
    assert not missing


def test_qwen3_dense_tiny_parity() -> None:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    hf_model = Qwen3ForCausalLM(config).eval()
    our_model = QwenTritonForCausalLM.from_config(config).eval()
    _assert_state_loads(our_model, hf_model)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        our_logits = our_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    torch.testing.assert_close(our_logits, hf_logits, atol=1e-5, rtol=1e-4)


def test_qwen3_dense_tiny_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    hf_model = Qwen3ForCausalLM(config).eval()
    our_model = QwenTritonForCausalLM.from_config(config).eval()
    _assert_state_loads(our_model, hf_model)

    prompt_ids = torch.randint(0, config.vocab_size, (1, 5))
    next_ids = torch.randint(0, config.vocab_size, (1, 1))
    prompt_mask = torch.ones_like(prompt_ids)
    full_ids = torch.cat((prompt_ids, next_ids), dim=-1)
    full_mask = torch.ones_like(full_ids)

    with torch.no_grad():
        prompt_out = our_model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True)
        prompt_cache_len = prompt_out.past_key_values.get_seq_length()
        cached_out = our_model(
            input_ids=next_ids,
            attention_mask=full_mask,
            past_key_values=prompt_out.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        full_out = our_model(input_ids=full_ids, attention_mask=full_mask, use_cache=False, logits_to_keep=1)

    torch.testing.assert_close(cached_out.logits, full_out.logits, atol=1e-5, rtol=1e-4)
    assert prompt_cache_len == prompt_ids.shape[1]
    assert cached_out.past_key_values.get_seq_length() == full_ids.shape[1]
    assert cached_out.past_key_values.key_cache[0].shape[-2] >= full_ids.shape[1]


def test_qwen3_moe_tiny_parity() -> None:
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        decoder_sparse_step=1,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[],
        tie_word_embeddings=False,
    )
    hf_model = Qwen3MoeForCausalLM(config).eval()
    our_model = QwenTritonForCausalLM.from_config(config).eval()
    _assert_state_loads(our_model, hf_model)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        our_logits = our_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    torch.testing.assert_close(our_logits, hf_logits, atol=1e-5, rtol=1e-4)


def test_qwen3_next_tiny_parity() -> None:
    torch.manual_seed(0)
    config = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=64,
        decoder_sparse_step=1,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=24,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        tie_word_embeddings=False,
    )
    hf_model = Qwen3NextForCausalLM(config).eval()
    our_model = QwenTritonForCausalLM.from_config(config).eval()
    _assert_state_loads(our_model, hf_model)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        our_logits = our_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    torch.testing.assert_close(our_logits, hf_logits, atol=2e-4, rtol=2e-4)
