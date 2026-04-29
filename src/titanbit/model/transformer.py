"""
titanbit.model.transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full BitNet b1.58 transformer language model.

Architecture mirrors LLaMA / Mistral with all nn.Linear replaced by
BitLinear (ternary quantised).  Non-linear components (embeddings,
norms, RoPE) remain in full precision.

Key components:
    - RoPE (Rotary Position Embeddings)
    - Group Query Attention (GQA) — optional
    - SwiGLU MLP
    - RMSNorm (pre-norm architecture)
    - BitLinear in all projection layers

Memory breakdown for 1.3B model on L20 (48GB):
    Weights (BF16 shadow):     ~2.6 GB
    Optimizer (AdamW, 2×):     ~5.2 GB
    Gradients:                 ~2.6 GB
    Activations (seq=2048):    ~8-12 GB
    ─────────────────────────────────
    Total:                     ~18-22 GB  ← leaves headroom for batch size
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from titanbit.model.bitlinear import BitLinear, RMSNorm
from titanbit.model.config import BitNetConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (Su et al., 2021).

    Pre-computes the sin/cos tables and applies them as complex
    rotations to the Q and K tensors.  This provides relative position
    information without any learnable parameters.

    We cache the tables for the maximum sequence length to avoid
    recomputation.  The theta parameter controls the frequency base
    (10000 is standard; higher values improve length generalisation).
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Pre-compute frequency table
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute sin/cos cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half: [x1, x2] → [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K tensors."""
    # cos/sin shape: (seq_len, head_dim) → broadcast over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class BitNetAttention(nn.Module):
    """
    Multi-Head / Grouped-Query Attention with BitLinear projections.

    When num_kv_heads < num_heads, uses GQA (Ainslie et al., 2023)
    for reduced KV-cache memory at inference time.

    All Q/K/V/O projections are BitLinear (ternary quantised).
    """

    def __init__(self, config: BitNetConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Projections — all BitLinear (ternary)
        self.q_proj = BitLinear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = BitLinear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = BitLinear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = BitLinear(self.num_heads * self.head_dim, config.hidden_size)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_length,
            theta=config.rope_theta,
        )

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache for inference
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # GQA: repeat KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        # Try FlashAttention first, fall back to manual
        try:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None and past_kv is None,
            )
        except RuntimeError:
            # Manual attention fallback
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attention_mask is None and past_kv is None:
                # Causal mask
                causal_mask = torch.triu(
                    torch.ones(T, k.shape[2], dtype=torch.bool, device=x.device),
                    diagonal=1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            elif attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(attn_out)

        return out, new_kv


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class BitNetMLP(nn.Module):
    """
    SwiGLU MLP with BitLinear projections.

    SwiGLU (Shazeer, 2020):
        MLP(x) = (Swish(xW_gate) ⊙ xW_up) × W_down

    The gated variant gives ~5% perplexity improvement over standard
    GeLU at equivalent parameter count.  All three linear layers are
    BitLinear (ternary quantised).
    """

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        i = config.intermediate_size

        if config.mlp_type == "swiglu":
            self.gate_proj = BitLinear(h, i)
            self.up_proj = BitLinear(h, i)
            self.down_proj = BitLinear(i, h)
        else:
            self.up_proj = BitLinear(h, i)
            self.down_proj = BitLinear(i, h)
            self.gate_proj = None

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_type == "swiglu":
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            out = gate * up
        else:
            out = F.gelu(self.up_proj(x))

        out = self.down_proj(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class BitNetBlock(nn.Module):
    """
    Single transformer block: pre-norm → attention → residual → pre-norm → MLP → residual.

    Uses pre-norm (RMSNorm before each sub-layer) following the LLaMA
    architecture, which is more stable than post-norm for deep models.
    """

    def __init__(self, config: BitNetConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = BitNetAttention(config, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = BitNetMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with residual
        residual = x
        x = self.attn_norm(x)
        attn_out, new_kv = self.attn(x, attention_mask, use_cache, past_kv)
        x = residual + attn_out

        # MLP with residual
        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        return x, new_kv


# ---------------------------------------------------------------------------
# Full transformer
# ---------------------------------------------------------------------------

class BitNetTransformer(nn.Module):
    """
    Complete BitNet b1.58 language model.

    Architecture summary:
        Embedding (full precision) → N × BitNetBlock → RMSNorm → LM Head

    All linear layers inside the blocks are BitLinear (ternary weights).
    The embedding and LM head remain in full precision because:
        1. Embeddings are already sparse lookups (not matmuls)
        2. The LM head's precision directly impacts output quality

    Usage
    -----
    >>> config = BitNetConfig(hidden_size=2048, num_layers=24, num_heads=32)
    >>> model = BitNetTransformer(config)
    >>> input_ids = torch.randint(0, 32000, (2, 512))
    >>> logits, loss = model(input_ids, labels=input_ids)
    """

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.config = config

        # Token + position embeddings (full precision)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.embedding_dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            BitNetBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # LM head (full precision — tied or untied)
        if config.tie_word_embeddings:
            self.lm_head = None  # will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        # Initialize weights
        self.apply(self._init_weights)

        # Log model size
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "BitNetTransformer initialised: %s params (%.1fM), %s trainable",
            f"{n_params:,}", n_params / 1e6, f"{n_trainable:,}",
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with scaled normal distribution."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BitLinear):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        input_ids      : (B, T) token indices
        labels         : (B, T) target token indices for loss computation
        attention_mask : optional attention mask
        use_cache      : whether to return KV cache
        past_kv        : list of (K, V) tuples from previous forward passes

        Returns
        -------
        logits : (B, T, vocab_size)
        loss   : scalar cross-entropy loss (if labels provided)
        """
        B, T = input_ids.shape

        # Embed tokens
        x = self.embed_tokens(input_ids)
        x = self.embed_dropout(x)

        # Run through transformer blocks
        new_kv_list = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kv[i] if past_kv is not None else None

            if self.gradient_checkpointing and self.training:
                x, new_kv = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, use_cache, layer_past_kv,
                    use_reentrant=False,
                )
            else:
                x, new_kv = layer(x, attention_mask, use_cache, layer_past_kv)

            if new_kv_list is not None:
                new_kv_list.append(new_kv)

        # Final norm
        x = self.norm(x)

        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        # Loss
        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory-efficient training."""
        self.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")

    def prepare_for_inference(self) -> None:
        """Pre-quantise all BitLinear weights for fast inference."""
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.prepare_for_inference()
        logger.info("Model prepared for inference (weights pre-quantised)")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters, optionally excluding embeddings."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n

    def estimate_mfu(self, batch_size: int, seq_len: int, dt: float) -> float:
        """
        Estimate Model Flops Utilisation (MFU).

        MFU = achieved_flops / peak_flops

        For a transformer, the FLOPs per forward pass ≈ 6 × N × B × T
        (where N = non-embedding params, B = batch, T = seq_len).
        Factor of 6: 2 for matmul, ×3 for forward+backward.

        Parameters
        ----------
        batch_size : micro batch size
        seq_len    : sequence length
        dt         : time per step in seconds

        Returns
        -------
        MFU as a fraction (0.0 to 1.0)
        """
        N = self.get_num_params(non_embedding=True)
        # 6N FLOPs per token (forward + backward)
        flops_per_token = 6 * N
        flops_per_step = flops_per_token * batch_size * seq_len
        flops_achieved = flops_per_step / dt

        # L20 peak: ~119 TFLOPS BF16 (Ada Lovelace)
        # Conservative estimate for sustained throughput
        flops_peak = 119.5e12  # BF16 tensor core peak

        mfu = flops_achieved / flops_peak
        return mfu
