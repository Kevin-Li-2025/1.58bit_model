"""
titanbit.model.config
~~~~~~~~~~~~~~~~~~~~~
Model configuration for BitNet b1.58 transformer.

Supports configurations from 125M to 3B+ parameters.
Pre-defined sizes follow Chinchilla-optimal scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class BitNetConfig:
    """
    Configuration for a BitNet b1.58 transformer.

    Key design decisions
    --------------------
    1. **RMSNorm before quantisation** — Following the BitNet paper, we apply
       RMSNorm (SubLN) before weight quantisation in every BitLinear layer.
       This stabilises the ternary rounding.

    2. **No bias** — Biases are removed from all linear layers.  The ternary
       weight constraint already limits expressivity; biases add parameters
       that don't benefit from the 1.58-bit compression.

    3. **SwiGLU MLP** — The gated variant gives ~5% perplexity improvement
       over standard GeLU at equivalent FLOPs (Shazeer, 2020).

    4. **RoPE** — Rotary positional embeddings for length generalisation.

    5. **Group Query Attention (GQA)** — Optional.  When num_kv_heads < num_heads,
       we use GQA for reduced KV-cache memory at inference.
    """

    # --- Architecture ---
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 32
    num_kv_heads: int | None = None    # None → MHA; < num_heads → GQA
    intermediate_size: int | None = None  # None → auto (8/3 × hidden, rounded to 256)
    vocab_size: int = 32000
    max_seq_length: int = 2048
    rope_theta: float = 10000.0

    # --- BitNet quantisation ---
    weight_bits: float = 1.58        # 1.58 = ternary {-1, 0, 1}
    activation_bits: int = 8         # AbsMean quantisation for activations
    use_ste: bool = True             # Straight-Through Estimator for gradients

    # --- Regularisation ---
    dropout: float = 0.0
    attention_dropout: float = 0.0
    embedding_dropout: float = 0.0

    # --- Normalisation ---
    norm_eps: float = 1e-5
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"

    # --- MLP ---
    mlp_type: Literal["swiglu", "gelu"] = "swiglu"

    # --- Misc ---
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        if self.intermediate_size is None:
            # SwiGLU optimal: 8/3 × hidden, rounded up to nearest 256
            raw = int(self.hidden_size * 8 / 3)
            self.intermediate_size = ((raw + 255) // 256) * 256

        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_params(self) -> int:
        """Estimate total parameter count (excluding embeddings)."""
        h = self.hidden_size
        L = self.num_layers
        V = self.vocab_size
        I = self.intermediate_size
        kv = self.num_kv_heads
        hd = self.head_dim

        # Embedding
        emb = V * h
        # Per-layer: attention (Q, K, V, O) + MLP + norms
        attn = h * (self.num_heads * hd) + h * (kv * hd) * 2 + (self.num_heads * hd) * h
        if self.mlp_type == "swiglu":
            mlp = h * I * 2 + I * h   # gate + up + down
        else:
            mlp = h * I + I * h
        norms = h * 2  # two RMSNorms per layer
        per_layer = attn + mlp + norms
        # Final norm + LM head
        final = h + (0 if self.tie_word_embeddings else V * h)
        return emb + L * per_layer + final

    @property
    def num_params_str(self) -> str:
        n = self.num_params
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        return f"{n / 1e6:.0f}M"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BitNetConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> BitNetConfig:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw.get("model", {}))

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# -----------------------------------------------------------------------
# Pre-defined model sizes
# -----------------------------------------------------------------------

BITNET_125M = BitNetConfig(
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    vocab_size=32000,
    max_seq_length=2048,
)

BITNET_350M = BitNetConfig(
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    vocab_size=32000,
    max_seq_length=2048,
)

BITNET_700M = BitNetConfig(
    hidden_size=1536,
    num_layers=24,
    num_heads=24,
    vocab_size=32000,
    max_seq_length=2048,
)

BITNET_1_3B = BitNetConfig(
    hidden_size=2048,
    num_layers=24,
    num_heads=32,
    vocab_size=32000,
    max_seq_length=2048,
)

BITNET_3B = BitNetConfig(
    hidden_size=3200,
    num_layers=26,
    num_heads=32,
    num_kv_heads=8,       # GQA for memory efficiency
    vocab_size=32000,
    max_seq_length=4096,
)

MODEL_REGISTRY: dict[str, BitNetConfig] = {
    "125M": BITNET_125M,
    "350M": BITNET_350M,
    "700M": BITNET_700M,
    "1.3B": BITNET_1_3B,
    "3B": BITNET_3B,
}
