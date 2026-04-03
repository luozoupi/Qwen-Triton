from __future__ import annotations

import torch
from torch import nn

from qwen_triton.kernels import silu_mul
from qwen_triton.modules.linear import TritonLinear


class QwenMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = TritonLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = TritonLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = TritonLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(silu_mul(gate, up))
