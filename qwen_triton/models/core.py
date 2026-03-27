from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from qwen_triton.configs import QwenTritonConfig
from qwen_triton.loaders import ensure_local_model_path, load_config_dict, load_hf_weights_into_model
from qwen_triton.modules import (
    Qwen3NextLinearAttention,
    QwenFullAttention,
    QwenMLP,
    QwenRMSNorm,
    QwenRotaryEmbedding,
    QwenSparseMoeBlock,
    QwenTritonCache,
)


def _parse_dtype(dtype: torch.dtype | str | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    lowered = dtype.lower()
    if lowered not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype}")
    return mapping[lowered]


def _build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    cache_position: torch.LongTensor,
    device: torch.device,
    past_key_values: QwenTritonCache | None,
    window_size: int | None = None,
) -> torch.Tensor:
    query_len = cache_position.shape[0]
    past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
    kv_len = past_seen_tokens + query_len
    key_positions = torch.arange(kv_len, device=device)
    causal = key_positions.unsqueeze(0) <= cache_position.unsqueeze(1)
    if window_size is not None:
        causal = causal & (key_positions.unsqueeze(0) > (cache_position.unsqueeze(1) - window_size))

    neg_inf = torch.finfo(torch.float32).min
    mask = torch.full((query_len, kv_len), neg_inf, device=device, dtype=torch.float32)
    mask = mask.masked_fill(causal, 0.0)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_len, kv_len)

    if attention_mask is not None:
        if attention_mask.shape[1] < kv_len:
            prefix = torch.ones((attention_mask.shape[0], kv_len - attention_mask.shape[1]), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((prefix, attention_mask.to(device)), dim=-1)
        key_padding = attention_mask[:, None, None, :kv_len].to(torch.float32)
        mask = mask + (1.0 - key_padding) * neg_inf
    return mask


def _layer_uses_moe(config: QwenTritonConfig, layer_idx: int) -> bool:
    if not config.is_moe or config.num_experts <= 0:
        return False
    if layer_idx in config.mlp_only_layers:
        return False
    return ((layer_idx + 1) % config.decoder_sparse_step) == 0


def _loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenTritonConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps, one_plus_weight=config.norm_one_plus)
        self.post_attention_layernorm = QwenRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            one_plus_weight=config.norm_one_plus,
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextLinearAttention(config, layer_idx)
        else:
            self.self_attn = QwenFullAttention(config, layer_idx)

        if _layer_uses_moe(config, layer_idx):
            self.mlp = QwenSparseMoeBlock(config)
        else:
            self.mlp = QwenMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: QwenTritonCache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        router_logits = None
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        hidden_states = residual + hidden_states
        return hidden_states, router_logits


class QwenTritonModel(nn.Module):
    def __init__(self, config: QwenTritonConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([QwenDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps, one_plus_weight=config.norm_one_plus)
        self.rotary_emb = QwenRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: QwenTritonCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        output_router_logits: bool = False,
        **_: Any,
    ) -> SimpleNamespace:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = QwenTritonCache(
                self.config.num_hidden_layers,
                self.config.layer_types,
                max_cache_len=self.config.max_position_embeddings,
            )

        if cache_position is None:
            past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        batch_size = inputs_embeds.shape[0]
        mask_map = {
            "full_attention": _build_causal_mask(
                attention_mask=attention_mask,
                batch_size=batch_size,
                cache_position=cache_position,
                device=inputs_embeds.device,
                past_key_values=past_key_values,
                window_size=None,
            ),
        }
        if "sliding_attention" in self.config.layer_types and self.config.sliding_window is not None:
            mask_map["sliding_attention"] = _build_causal_mask(
                attention_mask=attention_mask,
                batch_size=batch_size,
                cache_position=cache_position,
                device=inputs_embeds.device,
                past_key_values=past_key_values,
                window_size=self.config.sliding_window,
            )

        linear_attention_mask = attention_mask
        if self.config.is_qwen35_family:
            if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
                linear_attention_mask = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        router_logits: list[torch.Tensor] = []

        for decoder_layer in self.layers:
            layer_mask = linear_attention_mask if decoder_layer.layer_type == "linear_attention" else mask_map[decoder_layer.layer_type]
            hidden_states, layer_router_logits = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            if output_router_logits and layer_router_logits is not None:
                router_logits.append(layer_router_logits)

        hidden_states = self.norm(hidden_states)
        return SimpleNamespace(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            router_logits=tuple(router_logits) if output_router_logits else None,
        )


class QwenTritonForCausalLM(nn.Module):
    def __init__(self, config: QwenTritonConfig, backend: str = "triton", init_weights: bool = True) -> None:
        super().__init__()
        if backend != "triton":
            raise ValueError("Use from_reference_model for backend='ref'")
        self.config = config
        self.backend = backend
        self.ref_model = None
        self.model = QwenTritonModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if init_weights:
            self._reset_parameters()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    @classmethod
    def from_reference_model(cls, reference_model: nn.Module, config: QwenTritonConfig) -> "QwenTritonForCausalLM":
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.config = config
        instance.backend = "ref"
        instance.ref_model = reference_model
        return instance

    @classmethod
    def from_config(
        cls,
        config: QwenTritonConfig | dict | Any,
        backend: str = "triton",
        init_weights: bool = True,
    ) -> "QwenTritonForCausalLM":
        normalized = config if isinstance(config, QwenTritonConfig) else QwenTritonConfig.from_hf_config(config)
        if backend != "triton":
            raise ValueError("from_config currently supports backend='triton' only")
        return cls(normalized, backend=backend, init_weights=init_weights)

    @classmethod
    def from_pretrained_hf(
        cls,
        model_id: str,
        backend: str = "triton",
        device: str | torch.device = "cpu",
        dtype: torch.dtype | str | None = None,
    ) -> "QwenTritonForCausalLM":
        normalized = QwenTritonConfig.from_hf_config(load_config_dict(model_id))
        parsed_dtype = _parse_dtype(dtype) or torch.float32
        if backend == "ref":
            reference_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=parsed_dtype,
            )
            return cls.from_reference_model(reference_model.to(device), normalized)

        snapshot_path = ensure_local_model_path(model_id, include_weights=True)
        instance = cls.from_config(normalized, backend="triton", init_weights=False)
        instance.to(dtype=parsed_dtype)
        load_hf_weights_into_model(instance, snapshot_path)
        instance.to(device=device)
        instance.eval()
        return instance

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: QwenTritonCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_router_logits: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if self.backend == "ref":
            return self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif not isinstance(logits_to_keep, int):
            hidden_states = hidden_states[:, logits_to_keep, :]
        logits = self.lm_head(hidden_states)
        loss = _loss_fn(logits, labels) if labels is not None else None
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )
        if output_router_logits:
            result["router_logits"] = outputs.router_logits
        return result

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 16,
    ) -> torch.LongTensor:
        if self.backend == "ref" and hasattr(self.ref_model, "generate"):
            return self.ref_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = input_ids
        running_mask = attention_mask
        cache = None
        for _ in range(max_new_tokens):
            model_inputs = generated if cache is None else generated[:, -1:]
            outputs = self(
                input_ids=model_inputs,
                attention_mask=running_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
            if running_mask is not None:
                running_mask = torch.cat((running_mask, torch.ones_like(next_token, dtype=running_mask.dtype)), dim=-1)
        return generated
