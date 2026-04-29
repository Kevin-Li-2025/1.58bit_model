"""
titanbit.serving.speculative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Speculative Decoding for BitNet b1.58 models.

Speculative decoding (Leviathan et al., 2022; Chen et al., 2023)
accelerates autoregressive generation by using a small "draft" model
to predict K tokens ahead, then verifying them in parallel with the
larger "target" model.

Why this is especially powerful for BitNet:
    1. The draft model can be TINY (125M ternary = ~15MB in packed form)
    2. Ternary matmul is already fast → the draft model runs nearly free
    3. The acceptance rate is higher than FP16 because ternary models
       have a more "concentrated" output distribution

Algorithm:
    1. Draft model generates K candidate tokens autoregressively
    2. Target model processes all K+1 tokens in a single forward pass
    3. For each position i ∈ [1, K]:
        - Compute acceptance probability: min(1, p_target(x_i) / p_draft(x_i))
        - If accepted: keep x_i and continue
        - If rejected: resample from adjusted distribution and stop
    4. Bonus: always sample one additional token from the target model

Expected speedup: 1.5-3× depending on draft model quality and K.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from titanbit.model.transformer import BitNetTransformer
from titanbit.model.config import BitNetConfig
from titanbit.serving.engine import GenerationConfig, Sampler

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # Number of draft tokens to generate per speculation step
    num_speculative_tokens: int = 5

    # Draft model settings
    draft_model_path: str = ""           # path to draft checkpoint
    draft_temperature: float = 0.7       # sampling temperature for draft

    # Verification settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

    # Performance monitoring
    log_acceptance_rate: bool = True


class SpeculativeDecoder:
    """
    Speculative decoding engine using a draft + target model pair.

    The draft model is a smaller BitNet model (e.g., 125M) that
    generates candidate tokens quickly.  The target model (e.g., 1.3B)
    verifies them in parallel.

    Usage
    -----
    >>> draft = BitNetTransformer(BITNET_125M)
    >>> target = BitNetTransformer(BITNET_1_3B)
    >>> decoder = SpeculativeDecoder(target, draft)
    >>> tokens = decoder.generate(prompt_ids, max_new_tokens=256)

    Performance
    -----------
    On L20 GPU, typical speedups for BitNet models:
        K=3:  ~1.5× speedup
        K=5:  ~2.0× speedup
        K=8:  ~2.5× speedup (if acceptance rate > 70%)
    """

    def __init__(
        self,
        target_model: BitNetTransformer,
        draft_model: BitNetTransformer,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Target (large) model
        self.target = target_model.to(self.device).to(dtype)
        self.target.eval()
        self.target.prepare_for_inference()

        # Draft (small) model
        self.draft = draft_model.to(self.device).to(dtype)
        self.draft.eval()
        self.draft.prepare_for_inference()

        self.sampler = Sampler()

        # Statistics
        self._total_accepted = 0
        self._total_drafted = 0
        self._total_tokens_generated = 0
        self._total_target_calls = 0

        target_params = sum(p.numel() for p in self.target.parameters())
        draft_params = sum(p.numel() for p in self.draft.parameters())
        logger.info(
            "SpeculativeDecoder: target=%s, draft=%s (%.1f%% size)",
            f"{target_params:,}", f"{draft_params:,}",
            draft_params / target_params * 100,
        )

    @classmethod
    def from_checkpoints(
        cls,
        target_path: str,
        draft_path: str,
        device: str = "auto",
    ) -> SpeculativeDecoder:
        """Load both models from checkpoints."""
        dev = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else device if device != "auto" else "cpu"
        )

        # Load target
        t_ckpt = torch.load(target_path, map_location=dev, weights_only=False)
        t_cfg = BitNetConfig.from_dict(t_ckpt["model_config"])
        target = BitNetTransformer(t_cfg)
        target.load_state_dict(t_ckpt["model_state_dict"])

        # Load draft
        d_ckpt = torch.load(draft_path, map_location=dev, weights_only=False)
        d_cfg = BitNetConfig.from_dict(d_ckpt["model_config"])
        draft = BitNetTransformer(d_cfg)
        draft.load_state_dict(d_ckpt["model_state_dict"])

        return cls(target, draft, device=dev)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: SpeculativeConfig | None = None,
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.

        Algorithm for each step:
            1. Draft model: generate K tokens autoregressively
            2. Target model: single forward pass on all K+1 positions
            3. Verify: accept/reject each draft token
            4. Output: accepted tokens + 1 bonus token from target
        """
        if config is None:
            config = SpeculativeConfig()

        K = config.num_speculative_tokens
        input_ids = input_ids.to(self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids.clone()
        B = generated.shape[0]
        assert B == 1, "Speculative decoding currently supports batch_size=1"

        gen_config = GenerationConfig(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=True,
        )

        draft_gen_config = GenerationConfig(
            temperature=config.draft_temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=True,
        )

        start_time = time.monotonic()
        tokens_generated = 0

        while tokens_generated < config.max_new_tokens:
            # --- Step 1: Draft K tokens ---
            draft_tokens = []
            draft_probs_list = []
            draft_input = generated.clone()

            for _ in range(K):
                draft_logits, _ = self.draft(draft_input)
                draft_next_logits = draft_logits[:, -1, :]
                draft_probs = F.softmax(draft_next_logits / max(config.draft_temperature, 1e-8), dim=-1)

                # Sample from draft
                next_token = self.sampler.sample(draft_next_logits, draft_gen_config)
                draft_tokens.append(next_token)
                draft_probs_list.append(draft_probs)

                draft_input = torch.cat([draft_input, next_token.unsqueeze(-1)], dim=-1)

            self._total_drafted += K

            # --- Step 2: Target verifies all K tokens in parallel ---
            # Build the full sequence: original + K draft tokens
            candidate_ids = torch.cat(
                [generated] + [t.unsqueeze(-1) for t in draft_tokens],
                dim=-1,
            )

            target_logits, _ = self.target(candidate_ids)
            self._total_target_calls += 1

            # --- Step 3: Accept/reject each draft token ---
            n_accepted = 0
            current_pos = generated.shape[1] - 1  # position of last real token

            for i in range(K):
                target_pos = current_pos + i
                target_probs = F.softmax(
                    target_logits[:, target_pos, :] / max(config.temperature, 1e-8),
                    dim=-1,
                )

                draft_token = draft_tokens[i]
                draft_prob = draft_probs_list[i]

                # Acceptance probability: min(1, p_target / p_draft)
                p_target = target_probs[0, draft_token[0]].item()
                p_draft = draft_prob[0, draft_token[0]].item()

                acceptance_prob = min(1.0, p_target / max(p_draft, 1e-10))

                if torch.rand(1).item() < acceptance_prob:
                    # Accept this token
                    n_accepted += 1
                    generated = torch.cat([generated, draft_token.unsqueeze(-1)], dim=-1)
                    tokens_generated += 1

                    if tokens_generated >= config.max_new_tokens:
                        break
                else:
                    # Reject: resample from adjusted distribution
                    # p_adjusted = max(0, p_target - p_draft) normalised
                    adjusted = torch.clamp(target_probs - draft_prob, min=0)
                    adjusted_sum = adjusted.sum()
                    if adjusted_sum > 0:
                        adjusted = adjusted / adjusted_sum
                        resampled = torch.multinomial(adjusted, num_samples=1)
                    else:
                        resampled = torch.multinomial(target_probs, num_samples=1)

                    generated = torch.cat([generated, resampled], dim=-1)
                    tokens_generated += 1
                    break

            self._total_accepted += n_accepted

            # --- Step 4: Bonus token from target (if all K accepted) ---
            if n_accepted == K and tokens_generated < config.max_new_tokens:
                bonus_pos = current_pos + K
                if bonus_pos < target_logits.shape[1]:
                    bonus_logits = target_logits[:, bonus_pos, :]
                    bonus_token = self.sampler.sample(bonus_logits, gen_config)
                    generated = torch.cat([generated, bonus_token.unsqueeze(-1)], dim=-1)
                    tokens_generated += 1

        elapsed = time.monotonic() - start_time
        self._total_tokens_generated += tokens_generated

        if config.log_acceptance_rate:
            logger.info(
                "Speculative: %d tokens in %.2fs (%.1f tok/s) | "
                "acceptance=%.1f%% | avg_accepted=%.1f/%d",
                tokens_generated, elapsed,
                tokens_generated / max(elapsed, 1e-6),
                self.acceptance_rate * 100,
                self._total_accepted / max(self._total_target_calls, 1),
                K,
            )

        return generated

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate across all generation calls."""
        if self._total_drafted == 0:
            return 0.0
        return self._total_accepted / self._total_drafted

    @property
    def stats(self) -> dict[str, float]:
        return {
            "total_tokens": self._total_tokens_generated,
            "total_target_calls": self._total_target_calls,
            "total_drafted": self._total_drafted,
            "total_accepted": self._total_accepted,
            "acceptance_rate": self.acceptance_rate,
            "avg_accepted_per_step": (
                self._total_accepted / max(self._total_target_calls, 1)
            ),
        }
