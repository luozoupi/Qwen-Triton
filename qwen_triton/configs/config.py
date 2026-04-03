from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _default_layer_types(
    model_type: str,
    num_hidden_layers: int,
    use_sliding_window: bool,
    sliding_window: int | None,
    max_window_layers: int | None,
    full_attention_interval: int | None,
) -> list[str]:
    if model_type in {"qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"}:
        interval = full_attention_interval or 4
        return [
            "linear_attention" if bool((layer_idx + 1) % interval) else "full_attention"
            for layer_idx in range(num_hidden_layers)
        ]

    if use_sliding_window and sliding_window is not None:
        cutoff = max_window_layers if max_window_layers is not None else num_hidden_layers
        return [
            "sliding_attention" if layer_idx >= cutoff else "full_attention"
            for layer_idx in range(num_hidden_layers)
        ]
    return ["full_attention"] * num_hidden_layers


def _normalize_family(source_model_type: str, text_cfg: Mapping[str, Any]) -> str:
    num_experts = int(text_cfg.get("num_experts") or text_cfg.get("num_local_experts") or 0)
    shared_expert = int(text_cfg.get("shared_expert_intermediate_size", 0) or 0)

    if source_model_type == "qwen3":
        return "qwen3_dense"
    if source_model_type == "qwen3_moe":
        return "qwen3_moe"
    if source_model_type in {"qwen3_next", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}:
        return "qwen35_text_moe" if (num_experts > 0 or shared_expert > 0 or "moe" in source_model_type) else "qwen35_text_dense"
    raise ValueError(f"Unsupported Qwen model_type: {source_model_type}")


@dataclass
class QwenTritonConfig:
    family: str
    source_model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    torch_dtype: str | None = None

    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int | None = None
    layer_types: list[str] = field(default_factory=list)

    partial_rotary_factor: float = 1.0
    attn_output_gate: bool = False
    norm_one_plus: bool = False

    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0

    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    num_experts_per_tok: int = 0
    num_experts: int = 0
    norm_topk_prob: bool = False
    router_aux_loss_coef: float = 0.0
    mlp_only_layers: list[int] = field(default_factory=list)

    raw_config: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_moe(self) -> bool:
        return self.family in {"qwen3_moe", "qwen35_text_moe"}

    @property
    def is_qwen35_family(self) -> bool:
        return self.family in {"qwen35_text_dense", "qwen35_text_moe"}

    @property
    def rotary_dim(self) -> int:
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        rotary_dim = max(2, rotary_dim)
        if rotary_dim % 2:
            rotary_dim -= 1
        return rotary_dim

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "QwenTritonConfig":
        if hasattr(hf_config, "to_dict"):
            raw = hf_config.to_dict()
        elif isinstance(hf_config, Mapping):
            raw = dict(hf_config)
        else:
            raise TypeError(f"Unsupported config type: {type(hf_config)!r}")

        text_cfg = dict(raw.get("text_config") or raw)
        source_model_type = str(text_cfg.get("model_type") or raw.get("model_type"))
        family = _normalize_family(source_model_type, text_cfg)

        rope_params = dict(text_cfg.get("rope_parameters") or {})
        num_hidden_layers = int(text_cfg["num_hidden_layers"])
        use_sliding_window = bool(text_cfg.get("use_sliding_window", False))
        sliding_window = text_cfg.get("sliding_window")
        max_window_layers = text_cfg.get("max_window_layers")
        full_attention_interval = text_cfg.get("full_attention_interval")
        layer_types = list(text_cfg.get("layer_types") or _default_layer_types(
            model_type=source_model_type,
            num_hidden_layers=num_hidden_layers,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            full_attention_interval=full_attention_interval,
        ))

        head_dim = int(text_cfg.get("head_dim") or (int(text_cfg["hidden_size"]) // int(text_cfg["num_attention_heads"])))
        rope_theta = float(
            text_cfg.get("rope_theta")
            or rope_params.get("rope_theta")
            or 10000.0
        )
        partial_rotary_factor = float(text_cfg.get("partial_rotary_factor") or rope_params.get("partial_rotary_factor") or 1.0)
        attn_output_gate = text_cfg.get("attn_output_gate")
        if attn_output_gate is None and source_model_type in {"qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"}:
            attn_output_gate = True

        config = cls(
            family=family,
            source_model_type=source_model_type,
            vocab_size=int(text_cfg["vocab_size"]),
            hidden_size=int(text_cfg["hidden_size"]),
            intermediate_size=int(text_cfg.get("intermediate_size", 0) or 0),
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=int(text_cfg["num_attention_heads"]),
            num_key_value_heads=int(text_cfg.get("num_key_value_heads") or text_cfg["num_attention_heads"]),
            head_dim=head_dim,
            hidden_act=str(text_cfg.get("hidden_act", "silu")),
            max_position_embeddings=int(text_cfg.get("max_position_embeddings", 32768)),
            initializer_range=float(text_cfg.get("initializer_range", 0.02)),
            rms_norm_eps=float(text_cfg.get("rms_norm_eps", 1e-6)),
            use_cache=bool(text_cfg.get("use_cache", True)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings", False))),
            rope_theta=rope_theta,
            attention_bias=bool(text_cfg.get("attention_bias", False)),
            attention_dropout=float(text_cfg.get("attention_dropout", 0.0)),
            pad_token_id=text_cfg.get("pad_token_id", raw.get("pad_token_id")),
            bos_token_id=text_cfg.get("bos_token_id", raw.get("bos_token_id")),
            eos_token_id=text_cfg.get("eos_token_id", raw.get("eos_token_id")),
            torch_dtype=str(text_cfg.get("torch_dtype") or text_cfg.get("dtype") or raw.get("torch_dtype") or ""),
            use_sliding_window=use_sliding_window,
            sliding_window=int(sliding_window) if sliding_window is not None else None,
            max_window_layers=int(max_window_layers) if max_window_layers is not None else None,
            layer_types=layer_types,
            partial_rotary_factor=partial_rotary_factor,
            attn_output_gate=bool(attn_output_gate),
            norm_one_plus=source_model_type in {"qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"},
            linear_conv_kernel_dim=int(text_cfg.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(text_cfg.get("linear_key_head_dim", head_dim)),
            linear_value_head_dim=int(text_cfg.get("linear_value_head_dim", head_dim)),
            linear_num_key_heads=int(text_cfg.get("linear_num_key_heads", 0) or 0),
            linear_num_value_heads=int(text_cfg.get("linear_num_value_heads", 0) or 0),
            decoder_sparse_step=int(text_cfg.get("decoder_sparse_step", 1)),
            moe_intermediate_size=int(text_cfg.get("moe_intermediate_size", 0) or 0),
            shared_expert_intermediate_size=int(text_cfg.get("shared_expert_intermediate_size", 0) or 0),
            num_experts_per_tok=int(text_cfg.get("num_experts_per_tok", 0) or 0),
            num_experts=int(text_cfg.get("num_experts") or text_cfg.get("num_local_experts") or 0),
            norm_topk_prob=bool(text_cfg.get("norm_topk_prob", False)),
            router_aux_loss_coef=float(text_cfg.get("router_aux_loss_coef", 0.0)),
            mlp_only_layers=list(text_cfg.get("mlp_only_layers") or []),
            raw_config=raw,
        )

        if config.family == "qwen35_text_dense":
            config.num_experts = 0
            config.num_experts_per_tok = 0
            config.shared_expert_intermediate_size = 0
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
