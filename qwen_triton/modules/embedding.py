from __future__ import annotations

import torch
from torch import nn

from qwen_triton.kernels.embedding import triton_embedding


class TritonEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return triton_embedding(input_ids, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}"
