from __future__ import annotations

from dataclasses import dataclass

import torch

from qwen_triton.kernels import append_attention_kv, write_attention_kv


@dataclass
class QwenTritonCache:
    num_layers: int
    layer_types: list[str]
    max_cache_len: int | None = None

    def __post_init__(self) -> None:
        self.key_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.conv_states: list[torch.Tensor | None] = [None] * self.num_layers
        self.recurrent_states: list[torch.Tensor | None] = [None] * self.num_layers
        self.attention_lengths: list[int] = [0] * self.num_layers

    def _resolve_capacity(self, required_len: int) -> int:
        if required_len <= 0:
            return 16
        if self.max_cache_len is not None and required_len > self.max_cache_len:
            raise ValueError(f"Requested cache length {required_len} exceeds max_cache_len={self.max_cache_len}")
        capacity = max(16, 1 << (required_len - 1).bit_length())
        if self.max_cache_len is not None:
            capacity = min(capacity, self.max_cache_len)
        return max(required_len, capacity)

    def _ensure_attention_storage(
        self,
        storage: torch.Tensor | None,
        like: torch.Tensor,
        current_length: int,
        required_len: int,
    ) -> torch.Tensor:
        target_shape = (like.shape[0], like.shape[1], required_len, like.shape[3])
        if (
            storage is not None
            and storage.device == like.device
            and storage.dtype == like.dtype
            and storage.shape[0] == like.shape[0]
            and storage.shape[1] == like.shape[1]
            and storage.shape[3] == like.shape[3]
            and storage.shape[2] >= required_len
        ):
            return storage

        capacity = self._resolve_capacity(required_len)
        new_storage = like.new_empty((target_shape[0], target_shape[1], capacity, target_shape[3]))
        if (
            storage is not None
            and current_length > 0
            and storage.device == like.device
            and storage.dtype == like.dtype
            and storage.shape[0] == like.shape[0]
            and storage.shape[1] == like.shape[1]
            and storage.shape[3] == like.shape[3]
        ):
            copy_len = min(current_length, storage.shape[2], capacity)
            new_storage[:, :, :copy_len, :].copy_(storage[:, :, :copy_len, :])
        return new_storage

    def update_attention(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_position is None:
            self.key_cache[layer_idx] = append_attention_kv(self.key_cache[layer_idx], key_states)
            self.value_cache[layer_idx] = append_attention_kv(self.value_cache[layer_idx], value_states)
            self.attention_lengths[layer_idx] = int(self.key_cache[layer_idx].shape[-2])
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        positions = cache_position.to(device=key_states.device, dtype=torch.long).reshape(-1)
        if positions.numel() != key_states.shape[-2]:
            raise ValueError("cache_position length must match the sequence dimension of key/value states")

        required_len = int(positions.max().item()) + 1 if positions.numel() > 0 else self.attention_lengths[layer_idx]
        current_length = self.attention_lengths[layer_idx]
        self.key_cache[layer_idx] = self._ensure_attention_storage(
            self.key_cache[layer_idx],
            key_states,
            current_length=current_length,
            required_len=required_len,
        )
        self.value_cache[layer_idx] = self._ensure_attention_storage(
            self.value_cache[layer_idx],
            value_states,
            current_length=current_length,
            required_len=required_len,
        )
        write_attention_kv(self.key_cache[layer_idx], key_states, positions)
        write_attention_kv(self.value_cache[layer_idx], value_states, positions)
        self.attention_lengths[layer_idx] = max(current_length, required_len)
        valid_len = self.attention_lengths[layer_idx]
        return self.key_cache[layer_idx][:, :, :valid_len, :], self.value_cache[layer_idx][:, :, :valid_len, :]

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None and self.key_cache[layer_idx] is not None:
            return self.attention_lengths[layer_idx]
        for layer_idx, key_cache in enumerate(self.key_cache):
            if key_cache is not None:
                return self.attention_lengths[layer_idx]
        return 0

    @property
    def has_previous_state(self) -> bool:
        linear_indices = [idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"]
        if not linear_indices:
            return False
        return self.recurrent_states[linear_indices[-1]] is not None
