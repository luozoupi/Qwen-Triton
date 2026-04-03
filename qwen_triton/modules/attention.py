from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from qwen_triton.configs import QwenTritonConfig
from qwen_triton.kernels import sigmoid_mul
from qwen_triton.kernels.flash_attention import triton_flash_attention
from qwen_triton.modules.cache import QwenTritonCache
from qwen_triton.modules.linear import TritonLinear
from qwen_triton.modules.norms import QwenRMSNorm
from qwen_triton.modules.rotary import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class QwenFullAttention(nn.Module):
    def __init__(self, config: QwenTritonConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        q_out_dim = config.num_attention_heads * self.head_dim * (2 if config.attn_output_gate else 1)

        self.q_proj = TritonLinear(config.hidden_size, q_out_dim, bias=config.attention_bias)
        self.k_proj = TritonLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = TritonLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = TritonLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps, one_plus_weight=config.norm_one_plus)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps, one_plus_weight=config.norm_one_plus)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[QwenTritonCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        mixed_q = self.q_proj(hidden_states)

        gate = None
        if self.config.attn_output_gate:
            query_states, gate = torch.chunk(mixed_q.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
            query_states = query_states.view(hidden_shape)
        else:
            query_states = mixed_q.view(hidden_shape)

        key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(*input_shape, -1, self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update_attention(
                self.layer_idx,
                key_states,
                value_states,
                cache_position=cache_position,
            )

        # Use Triton flash attention with native GQA support
        # No need for repeat_kv or fp32 upcasting — flash attention handles both
        is_causal = attention_mask is None and query_states.shape[2] == key_states.shape[2]
        if is_causal:
            attn_output = triton_flash_attention(
                query_states,
                key_states,
                value_states,
                scale=self.scaling,
                is_causal=True,
                num_kv_groups=self.num_key_value_groups,
            )
        else:
            # Fallback for cached decode or when mask is provided:
            # Flash attention causal mode requires Q_len == K_len.
            # For decode (Q_len=1), causal masking is unnecessary.
            # For masked cases, expand KV and use SDPA.
            if query_states.shape[2] == 1:
                # Single-token decode: no causal mask needed, just dot product
                attn_output = triton_flash_attention(
                    query_states,
                    key_states,
                    value_states,
                    scale=self.scaling,
                    is_causal=False,
                    num_kv_groups=self.num_key_value_groups,
                )
            else:
                # Fallback to SDPA for complex mask scenarios
                key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
                value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)
                attn_mask = None if attention_mask is None else attention_mask[:, :, :, : key_states_expanded.shape[-2]]
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states.to(torch.float32),
                    key_states_expanded.to(torch.float32),
                    value_states_expanded.to(torch.float32),
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scaling,
                ).to(hidden_states.dtype)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        if gate is not None:
            attn_output = sigmoid_mul(attn_output, gate)
        return self.o_proj(attn_output)
