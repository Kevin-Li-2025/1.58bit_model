"""
titanbit.alignment.dpo
~~~~~~~~~~~~~~~~~~~~~~
Direct Preference Optimisation (DPO) for BitNet b1.58.

DPO (Rafailov et al., 2023) eliminates the need for a separate reward
model by directly optimising the policy against preference pairs:

    L_DPO = -E[log σ(β (log π_θ(y_w|x) - log π_ref(y_w|x)
                        - log π_θ(y_l|x) + log π_ref(y_l|x)))]

where:
    π_θ   = current policy (being trained)
    π_ref = reference policy (frozen copy of the SFT model)
    y_w   = chosen (winning) response
    y_l   = rejected (losing) response
    β     = temperature parameter controlling deviation from reference

Research Question
-----------------
Can ternary-weight models be effectively aligned via DPO?

The hypothesis is that the quantised weight manifold creates a
"rougher" loss landscape for preference learning, requiring:
    1. Lower β (more conservative updates)
    2. Higher gradient clipping thresholds
    3. Careful handling of the STE interaction with DPO gradients

This module implements DPO from scratch to study these dynamics.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from titanbit.model.transformer import BitNetTransformer
from titanbit.model.config import BitNetConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    """DPO training configuration."""

    # DPO hyperparameters
    beta: float = 0.1              # KL penalty coefficient
    label_smoothing: float = 0.0   # label smoothing for DPO loss
    loss_type: str = "sigmoid"     # "sigmoid" (standard DPO) | "hinge" | "ipo"

    # Optimisation
    learning_rate: float = 5e-7    # much lower than pretraining
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 5000

    # Batching
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_length: int = 1024         # max sequence length
    max_prompt_length: int = 512

    # Precision
    dtype: str = "bfloat16"

    # Logging & saving
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 200
    checkpoint_dir: str = "./checkpoints/dpo"

    # Reference model
    reference_free: bool = False   # if True, skip reference model (SimPO-style)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DPOConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------

class DPOTrainer:
    """
    Direct Preference Optimisation trainer for BitNet models.

    This implements DPO from scratch — no dependency on trl or
    other alignment libraries.  This gives us full control over
    the interaction between DPO gradients and the STE quantisation.

    Usage
    -----
    >>> from titanbit.alignment import DPOTrainer, DPOConfig
    >>> config = DPOConfig(beta=0.1, learning_rate=5e-7)
    >>> trainer = DPOTrainer(model, config, tokenizer)
    >>> trainer.train(preference_dataloader)
    """

    def __init__(
        self,
        model: BitNetTransformer,
        config: DPOConfig,
        tokenizer: Any,
        ref_model: Optional[BitNetTransformer] = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Active policy
        self.model = model.to(self.device)

        # Reference policy (frozen)
        if config.reference_free:
            self.ref_model = None
            logger.info("Reference-free mode (SimPO-style)")
        elif ref_model is not None:
            self.ref_model = ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            # Deep copy the model as the reference
            logger.info("Creating reference model (deep copy)...")
            self.ref_model = copy.deepcopy(model).to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        # Precision
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.dtype]

        self.amp_ctx = torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        )

        # Optimizer — only tune the active model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Tracking
        self.global_step = 0
        self._metrics: dict[str, list[float]] = {
            "loss": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_margin": [],
            "accuracy": [],
        }

    # ----- Core DPO Loss -----

    def compute_logprobs(
        self,
        model: BitNetTransformer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log-probabilities for the given sequences.

        This is the critical function for DPO.  We must be careful
        to compute log-probs only on the completion tokens (where
        labels != -100).

        Returns
        -------
        Sum of log-probabilities over completion tokens: shape (B,)
        """
        with self.amp_ctx:
            logits, _ = model(input_ids)

        # Shift: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = (shift_labels != -100)

        # Per-token log probabilities
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)

        # Gather log-probs for the actual tokens
        per_token_logps = torch.gather(
            log_probs, dim=-1,
            index=shift_labels.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)

        # Mask out prompt tokens and padding
        per_token_logps = per_token_logps * shift_mask.float()

        # Sum over sequence (per-example total log-prob)
        return per_token_logps.sum(dim=-1)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the DPO loss.

        DPO Loss = -log σ(β × (Δ_policy - Δ_reference))

        where Δ = log π(chosen) - log π(rejected)

        Parameters
        ----------
        policy_chosen_logps    : log π_θ(y_w|x)
        policy_rejected_logps  : log π_θ(y_l|x)
        reference_chosen_logps : log π_ref(y_w|x)
        reference_rejected_logps: log π_ref(y_l|x)

        Returns
        -------
        loss    : scalar DPO loss
        metrics : dict with reward margins and accuracy
        """
        beta = self.config.beta

        # Log-ratio differences
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        # The core DPO quantity
        logits_diff = beta * (pi_logratios - ref_logratios)

        # Implicit rewards
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        # Loss variants
        if self.config.loss_type == "sigmoid":
            # Standard DPO
            loss = -F.logsigmoid(logits_diff).mean()
        elif self.config.loss_type == "hinge":
            # Hinge-style DPO
            loss = torch.relu(1.0 - logits_diff).mean()
        elif self.config.loss_type == "ipo":
            # Identity Preference Optimisation (Azar et al., 2023)
            loss = ((logits_diff - 1.0 / (2.0 * beta)) ** 2).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        # Label smoothing
        if self.config.label_smoothing > 0:
            eps = self.config.label_smoothing
            loss = (1 - eps) * loss + eps * (-F.logsigmoid(-logits_diff).mean())

        # Metrics
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (logits_diff > 0).float().mean().item()

        metrics = {
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
        }

        return loss, metrics

    def dpo_loss_reference_free(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Reference-free DPO (SimPO-style).

        Skips the reference model entirely — useful when you want
        to save 50% of VRAM by not loading a frozen copy.

        Loss = -log σ(β × (log π(chosen) - log π(rejected)))
        """
        beta = self.config.beta
        logits_diff = beta * (policy_chosen_logps - policy_rejected_logps)
        loss = -F.logsigmoid(logits_diff).mean()

        accuracy = (logits_diff > 0).float().mean().item()
        metrics = {
            "chosen_reward": policy_chosen_logps.mean().item(),
            "rejected_reward": policy_rejected_logps.mean().item(),
            "reward_margin": (policy_chosen_logps - policy_rejected_logps).mean().item(),
            "accuracy": accuracy,
        }

        return loss, metrics

    # ----- Training Loop -----

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, dict[str, float]]:
        """Execute a single DPO training step."""
        self.model.train()

        # Move batch to device
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        # Compute policy log-probs
        policy_chosen_logps = self.compute_logprobs(
            self.model, chosen_ids, chosen_mask, chosen_labels
        )
        policy_rejected_logps = self.compute_logprobs(
            self.model, rejected_ids, rejected_mask, rejected_labels
        )

        # Compute loss
        if self.config.reference_free or self.ref_model is None:
            loss, metrics = self.dpo_loss_reference_free(
                policy_chosen_logps, policy_rejected_logps
            )
        else:
            # Compute reference log-probs (no gradient)
            with torch.no_grad():
                ref_chosen_logps = self.compute_logprobs(
                    self.ref_model, chosen_ids, chosen_mask, chosen_labels
                )
                ref_rejected_logps = self.compute_logprobs(
                    self.ref_model, rejected_ids, rejected_mask, rejected_labels
                )

            loss, metrics = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
            )

        # Scale for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        return loss.item(), metrics

    def train(self, dataloader: Any) -> dict[str, Any]:
        """
        Full DPO training loop.

        Returns
        -------
        Training statistics dict
        """
        cfg = self.config
        logger.info("=" * 60)
        logger.info("TitanBit DPO Alignment")
        logger.info("  beta=%.3f | lr=%.2e | loss=%s", cfg.beta, cfg.learning_rate, cfg.loss_type)
        logger.info("  batch=%d x %d = %d effective", cfg.batch_size, cfg.gradient_accumulation_steps, cfg.effective_batch_size)
        logger.info("  reference_free=%s", cfg.reference_free)
        logger.info("=" * 60)

        start_time = time.monotonic()
        data_iter = iter(dataloader)

        while self.global_step < cfg.max_steps:
            # LR warmup
            lr = self._get_lr(self.global_step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            total_loss = 0.0
            all_metrics: dict[str, list[float]] = {k: [] for k in self._metrics}

            for _ in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                step_loss, step_metrics = self.train_step(batch)
                total_loss += step_loss
                for k, v in step_metrics.items():
                    all_metrics[k].append(v)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            avg_loss = total_loss / cfg.gradient_accumulation_steps

            # Track metrics
            self._metrics["loss"].append(avg_loss)
            for k in all_metrics:
                if all_metrics[k]:
                    avg = sum(all_metrics[k]) / len(all_metrics[k])
                    self._metrics[k].append(avg)

            # Logging
            if self.global_step % cfg.log_interval == 0:
                acc = self._metrics["accuracy"][-1] if self._metrics["accuracy"] else 0
                margin = self._metrics["reward_margin"][-1] if self._metrics["reward_margin"] else 0
                logger.info(
                    "step=%d | loss=%.4f | acc=%.1f%% | margin=%.3f | lr=%.2e",
                    self.global_step, avg_loss, acc * 100, margin, lr,
                )

            # Save checkpoint
            if self.global_step % cfg.save_interval == 0:
                self._save_checkpoint()

        elapsed = time.monotonic() - start_time
        stats = {
            "total_steps": self.global_step,
            "elapsed_seconds": elapsed,
            "final_loss": self._metrics["loss"][-1] if self._metrics["loss"] else 0,
            "final_accuracy": self._metrics["accuracy"][-1] if self._metrics["accuracy"] else 0,
            "final_reward_margin": self._metrics["reward_margin"][-1] if self._metrics["reward_margin"] else 0,
        }

        logger.info("DPO training complete: %d steps in %.1f min", self.global_step, elapsed / 60)
        return stats

    def _get_lr(self, step: int) -> float:
        """Linear warmup then constant."""
        cfg = self.config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step / max(1, cfg.warmup_steps))
        return cfg.learning_rate

    def _save_checkpoint(self) -> None:
        """Save DPO checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.config.checkpoint_dir,
            f"dpo_step_{self.global_step:06d}.pt",
        )
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config.__dict__,
            "metrics": {k: v[-100:] for k, v in self._metrics.items()},
        }, path)
        logger.info("DPO checkpoint saved: %s", path)

    @property
    def metrics_summary(self) -> dict[str, float]:
        """Get the latest metrics."""
        return {
            k: v[-1] if v else 0.0
            for k, v in self._metrics.items()
        }
