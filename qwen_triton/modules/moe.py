from __future__ import annotations

import torch
from torch import nn

from qwen_triton.configs import QwenTritonConfig
from qwen_triton.kernels.moe_routing import triton_index_add, triton_one_hot, triton_topk
from qwen_triton.kernels.sigmoid_mul import sigmoid_mul
from qwen_triton.kernels.softmax import triton_softmax
from qwen_triton.modules.linear import TritonLinear
from qwen_triton.modules.mlp import QwenMLP


class QwenSparseMoeBlock(nn.Module):
    def __init__(self, config: QwenTritonConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = TritonLinear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [QwenMLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts)]
        )

        self.shared_expert = None
        self.shared_expert_gate = None
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = QwenMLP(config.hidden_size, config.shared_expert_intermediate_size)
            self.shared_expert_gate = TritonLinear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(flat_hidden)
        routing_weights = triton_softmax(router_logits.float())
        routing_weights, selected_experts = triton_topk(routing_weights, self.top_k)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat_hidden.dtype)

        final_hidden = torch.zeros_like(flat_hidden)
        expert_mask = triton_one_hot(selected_experts, self.num_experts).permute(2, 1, 0)
        expert_hit = torch.nonzero(expert_mask.sum(dim=(-1, -2)) > 0, as_tuple=False).view(-1)

        for expert_idx in expert_hit.tolist():
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_hidden = expert_layer(flat_hidden[top_x]) * routing_weights[top_x, idx, None]
            final_hidden = triton_index_add(final_hidden, current_hidden.to(flat_hidden.dtype), top_x)

        if self.shared_expert is not None and self.shared_expert_gate is not None:
            shared_hidden = self.shared_expert(flat_hidden)
            gate_value = self.shared_expert_gate(flat_hidden)
            shared_hidden = sigmoid_mul(shared_hidden, gate_value.expand_as(shared_hidden))
            final_hidden = final_hidden + shared_hidden

        return final_hidden.view(batch_size, sequence_length, hidden_dim), router_logits
