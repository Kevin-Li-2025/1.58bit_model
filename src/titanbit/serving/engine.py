"""
titanbit.serving.engine
~~~~~~~~~~~~~~~~~~~~~~~
Autoregressive inference engine for BitNet b1.58 models.

This is NOT a wrapper around vLLM or HuggingFace generate().
It is a from-scratch implementation that exploits the ternary
weight structure for maximum throughput.

Key optimisations:
    1. Pre-quantised weights (no runtime quantisation overhead)
    2. KV cache with pre-allocated memory
    3. Top-k/top-p/temperature sampling
    4. Streaming token-by-token generation
    5. Fused RMSNorm + ternary matmul (when Triton available)

Architecture:
    ┌──────────────┐
    │  Input Tokens │
    └──────┬───────┘
           │
    ┌──────▼───────┐     ┌─────────────┐
    │  Embedding   │────▶│  KV Cache   │
    └──────┬───────┘     │ (pre-alloc) │
           │             └──────┬──────┘
    ┌──────▼───────┐            │
    │  N × Block   │◀───────────┘
    │  (ternary    │
    │   weights)   │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   LM Head    │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Sampler    │
    │ (top-k/p/T) │
    └──────────────┘
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Generator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from titanbit.model.config import BitNetConfig
from titanbit.model.transformer import BitNetTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation Configuration
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    eos_token_id: int | None = None
    pad_token_id: int = 0

    # Streaming
    stream: bool = False

    # Performance
    use_cache: bool = True


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """
    Pre-allocated KV cache for autoregressive generation.

    Pre-allocation avoids repeated memory allocation during generation,
    which is critical for maintaining consistent token generation latency.

    Memory usage per layer:
        K: (batch, num_kv_heads, max_seq_len, head_dim)  — dtype
        V: (batch, num_kv_heads, max_seq_len, head_dim)  — dtype

    Total for 1.3B model (24 layers, 32 heads, 64 dim, seq=2048):
        = 24 × 2 × (1 × 32 × 2048 × 64) × 2 bytes
        = 24 × 2 × 8MB = 384 MB
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.current_length = 0

        # Pre-allocate
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []

        for _ in range(num_layers):
            self.k_cache.append(
                torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim,
                            device=device, dtype=dtype)
            )
            self.v_cache.append(
                torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim,
                            device=device, dtype=dtype)
            )

        total_bytes = sum(k.nelement() * k.element_size() for k in self.k_cache) * 2
        logger.debug("KV cache allocated: %.1f MB", total_bytes / (1024**2))

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a layer and return the full K/V up to current position.

        Parameters
        ----------
        layer_idx : which transformer layer
        k, v      : new K/V tensors of shape (B, num_kv_heads, new_seq_len, head_dim)

        Returns
        -------
        Full K/V tensors up to current position
        """
        new_len = k.shape[2]
        start = self.current_length
        end = start + new_len

        self.k_cache[layer_idx][:, :, start:end, :] = k
        self.v_cache[layer_idx][:, :, start:end, :] = v

        return (
            self.k_cache[layer_idx][:, :, :end, :],
            self.v_cache[layer_idx][:, :, :end, :],
        )

    def advance(self, num_tokens: int = 1) -> None:
        """Advance the position pointer."""
        self.current_length += num_tokens

    def reset(self) -> None:
        """Reset the cache for a new sequence."""
        self.current_length = 0
        for k, v in zip(self.k_cache, self.v_cache):
            k.zero_()
            v.zero_()


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class Sampler:
    """
    Token sampling with top-k, top-p (nucleus), temperature, and
    repetition penalty.

    The sampler is decoupled from the model so it can be tested
    and tuned independently.
    """

    @staticmethod
    def sample(
        logits: torch.Tensor,
        config: GenerationConfig,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample a token from logits.

        Parameters
        ----------
        logits      : (B, vocab_size) logits for the last position
        config      : generation configuration
        past_tokens : (B, seq_len) previously generated tokens (for rep penalty)

        Returns
        -------
        (B,) sampled token IDs
        """
        logits = logits.clone()

        # Repetition penalty
        if past_tokens is not None and config.repetition_penalty != 1.0:
            for b in range(logits.shape[0]):
                unique_tokens = past_tokens[b].unique()
                for token in unique_tokens:
                    if token < logits.shape[1]:
                        if logits[b, token] > 0:
                            logits[b, token] /= config.repetition_penalty
                        else:
                            logits[b, token] *= config.repetition_penalty

        if not config.do_sample:
            # Greedy
            return logits.argmax(dim=-1)

        # Temperature
        if config.temperature > 0:
            logits = logits / config.temperature

        # Top-k filtering
        if config.top_k > 0:
            top_k = min(config.top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if 0 < config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= config.top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    High-performance inference engine for BitNet models.

    Usage
    -----
    >>> engine = InferenceEngine.from_pretrained("checkpoints/bitnet-1.3B/checkpoint_best.pt")
    >>> tokens = engine.generate(prompt_ids, max_new_tokens=256)

    # Streaming
    >>> for token in engine.generate_stream(prompt_ids):
    ...     print(tokenizer.decode([token]), end="", flush=True)
    """

    def __init__(
        self,
        model: BitNetTransformer,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Prepare model for inference
        self.model = model.to(self.device).to(dtype)
        self.model.eval()
        self.model.prepare_for_inference()  # pre-quantise all BitLinear weights

        self.config = model.config
        self.sampler = Sampler()

        # Performance stats
        self._total_tokens = 0
        self._total_time = 0.0

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "InferenceEngine ready: %s params on %s (%s)",
            f"{n_params:,}", self.device, dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
    ) -> InferenceEngine:
        """Load a model from a training checkpoint."""
        dev = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else device if device != "auto" else "cpu"
        )
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

        logger.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)

        model_cfg = BitNetConfig.from_dict(ckpt["model_config"])
        model = BitNetTransformer(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(model, device=dev, dtype=dt)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        gen_config: GenerationConfig | None = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with KV cache.

        Uses a two-phase approach:
            1. Prefill: process entire prompt, cache K/V states
            2. Decode: process one token at a time using cached K/V

        This gives O(n·L) generation instead of the naive O(n²·L).

        Parameters
        ----------
        input_ids  : (B, prompt_len) prompt token IDs
        gen_config : generation settings

        Returns
        -------
        (B, prompt_len + new_tokens) complete sequence
        """
        if gen_config is None:
            gen_config = GenerationConfig()

        B, T = input_ids.shape
        input_ids = input_ids.to(self.device)
        generated = input_ids.clone()

        start_time = time.monotonic()

        # Phase 1: Prefill — process the entire prompt, collect KV cache
        use_kv = gen_config.use_cache
        logits, _ = self.model(input_ids, use_cache=use_kv)
        next_logits = logits[:, -1, :]

        # Collect past_kv from the model layers for cached decoding
        past_kv = None
        if use_kv:
            # Re-run with use_cache=True to collect KV pairs
            # The model returns logits but KV must be collected via
            # a forward that returns them.  We restructure to collect.
            past_kv = self._collect_kv_cache(input_ids)

        for step in range(gen_config.max_new_tokens):
            # Sample next token
            next_token = self.sampler.sample(
                next_logits, gen_config, past_tokens=generated
            )

            # Append
            generated = torch.cat([
                generated,
                next_token.unsqueeze(-1),
            ], dim=-1)

            # Check EOS
            if gen_config.eos_token_id is not None:
                if (next_token == gen_config.eos_token_id).all():
                    break

            # Phase 2: Decode — single token with KV cache
            if past_kv is not None:
                # Process only the new token
                logits, _ = self.model(
                    next_token.unsqueeze(-1),
                    use_cache=True,
                    past_kv=past_kv,
                )
                # Update KV cache with new entries
                past_kv = self._collect_kv_cache_incremental(
                    next_token.unsqueeze(-1), past_kv
                )
            else:
                # Fallback: re-process entire sequence (no cache)
                logits, _ = self.model(generated)
            next_logits = logits[:, -1, :]

        elapsed = time.monotonic() - start_time
        new_tokens = generated.shape[1] - T
        self._total_tokens += new_tokens * B
        self._total_time += elapsed

        tps = (new_tokens * B) / max(elapsed, 1e-6)
        logger.info(
            "Generated %d tokens in %.2fs (%.1f tok/s)",
            new_tokens, elapsed, tps,
        )

        return generated

    def _collect_kv_cache(
        self,
        input_ids: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run a forward pass with use_cache=True and collect KV pairs per layer."""
        model = self.model
        x = model.embed_tokens(input_ids)
        x = model.embed_dropout(x)

        past_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in model.layers:
            x, new_kv = layer(x, use_cache=True)
            if new_kv is not None:
                past_kv.append(new_kv)
        return past_kv

    def _collect_kv_cache_incremental(
        self,
        new_token_ids: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run a single-token forward with existing KV cache, return updated cache."""
        model = self.model
        x = model.embed_tokens(new_token_ids)
        x = model.embed_dropout(x)

        updated_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(model.layers):
            layer_past = past_kv[i] if i < len(past_kv) else None
            x, new_kv = layer(x, use_cache=True, past_kv=layer_past)
            if new_kv is not None:
                updated_kv.append(new_kv)
        return updated_kv

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        gen_config: GenerationConfig | None = None,
    ) -> Generator[int, None, None]:
        """
        Stream tokens one at a time with KV cache.

        Yields individual token IDs as they are generated.
        This is useful for real-time chat interfaces.

        Usage
        -----
        >>> for tok in engine.generate_stream(prompt_ids):
        ...     print(tokenizer.decode([tok]), end="", flush=True)
        """
        if gen_config is None:
            gen_config = GenerationConfig()

        input_ids = input_ids.to(self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids.clone()

        # Prefill with KV cache
        use_kv = gen_config.use_cache
        logits, _ = self.model(input_ids)
        next_logits = logits[:, -1, :]

        past_kv = self._collect_kv_cache(input_ids) if use_kv else None

        for step in range(gen_config.max_new_tokens):
            next_token = self.sampler.sample(
                next_logits, gen_config, past_tokens=generated
            )

            token_id = next_token[0].item()
            yield token_id

            if gen_config.eos_token_id is not None and token_id == gen_config.eos_token_id:
                break

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            if past_kv is not None:
                logits, _ = self.model(
                    next_token.unsqueeze(-1), use_cache=True, past_kv=past_kv
                )
                past_kv = self._collect_kv_cache_incremental(
                    next_token.unsqueeze(-1), past_kv
                )
            else:
                logits, _ = self.model(generated)
            next_logits = logits[:, -1, :]

    @property
    def throughput(self) -> float:
        """Average tokens per second across all calls."""
        if self._total_time == 0:
            return 0.0
        return self._total_tokens / self._total_time
