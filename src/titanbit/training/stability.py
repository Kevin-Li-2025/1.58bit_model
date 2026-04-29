"""
titanbit.training.stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loss spike detection and automatic recovery for BitNet training.

Problem
-------
Ternary quantisation introduces a non-smooth optimisation landscape.
The STE (Straight-Through Estimator) creates "shadow gradients" that
don't perfectly track the true loss surface.  This can cause:

    1. **Loss spikes** — sudden 5-50× increase in loss
    2. **Gradient explosions** — norm > 1000 in a single step
    3. **Oscillations** — loss bouncing between two basins

These issues are well-documented in QAT literature (Jacob et al., 2018)
and are more severe for extreme quantisation (1.58-bit).

Solution
--------
A multi-layered stability system:

    Layer 1: Gradient clipping (max_norm=1.0)
    Layer 2: Loss spike detection with exponential moving average
    Layer 3: Automatic rollback to last stable checkpoint
    Layer 4: Learning rate scaling after recovery

This is inspired by production training systems at Google (PaLM)
and Meta (LLaMA), where training stability at scale is a first-class
engineering concern.
"""

from __future__ import annotations

import copy
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class StabilityConfig:
    """Configuration for the stability system."""

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Loss spike detection
    spike_threshold: float = 5.0       # spike if loss > threshold × EMA
    ema_alpha: float = 0.99            # EMA smoothing factor
    min_steps_before_detection: int = 100  # warmup before spike detection

    # Recovery
    max_rollbacks: int = 3             # max rollbacks before giving up
    lr_scale_after_recovery: float = 0.5  # halve LR after rollback
    cooldown_steps: int = 50           # steps to wait before re-checking

    # Checkpointing
    stable_checkpoint_interval: int = 500  # save stable state every N steps

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StabilityConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StabilityMonitor:
    """
    Monitors training stability and handles recovery.

    Usage in training loop:
        monitor = StabilityMonitor(config, model, optimizer)

        for step, batch in enumerate(train_loader):
            loss = train_step(batch)
            grad_norm = monitor.clip_gradients(model)

            action = monitor.check(step, loss.item(), grad_norm)
            if action == "rollback":
                monitor.rollback(model, optimizer, scheduler)
                continue
            elif action == "stop":
                break

            if step % config.stable_checkpoint_interval == 0:
                monitor.save_stable_state(model, optimizer, scheduler)
    """

    def __init__(
        self,
        config: StabilityConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.config = config

        # EMA tracking
        self._loss_ema: Optional[float] = None
        self._grad_norm_ema: Optional[float] = None

        # Spike tracking
        self._spike_count = 0
        self._rollback_count = 0
        self._cooldown_until = 0

        # Loss history (last 100 values)
        self._loss_history: deque[float] = deque(maxlen=100)

        # Stable state (for rollback)
        self._stable_model_state: Optional[dict] = None
        self._stable_optimizer_state: Optional[dict] = None
        self._stable_step: int = 0

        # Initial snapshot
        self._save_state(model, optimizer)

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the pre-clip norm."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.config.max_grad_norm,
        )
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    def check(self, step: int, loss: float, grad_norm: float) -> str:
        """
        Check training stability after each step.

        Returns:
            "ok"       — training is stable
            "rollback" — loss spike detected, should rollback
            "stop"     — too many rollbacks, should stop training
        """
        self._loss_history.append(loss)

        # Update EMAs
        if self._loss_ema is None:
            self._loss_ema = loss
            self._grad_norm_ema = grad_norm
        else:
            a = self.config.ema_alpha
            self._loss_ema = a * self._loss_ema + (1 - a) * loss
            self._grad_norm_ema = a * self._grad_norm_ema + (1 - a) * grad_norm

        # Skip detection during warmup or cooldown
        if step < self.config.min_steps_before_detection:
            return "ok"
        if step < self._cooldown_until:
            return "ok"

        # --- Loss spike detection ---
        is_spike = False

        # Check 1: Loss exceeds threshold × EMA
        if loss > self.config.spike_threshold * self._loss_ema:
            logger.warning(
                "⚠️  Loss spike at step %d: %.4f (EMA: %.4f, ratio: %.1f×)",
                step, loss, self._loss_ema, loss / self._loss_ema,
            )
            is_spike = True

        # Check 2: Loss is NaN or Inf
        if not torch.isfinite(torch.tensor(loss)):
            logger.error("❌ Loss is NaN/Inf at step %d!", step)
            is_spike = True

        # Check 3: Gradient explosion
        if grad_norm > 100 * (self._grad_norm_ema or 1.0):
            logger.warning(
                "⚠️  Gradient explosion at step %d: %.2f (EMA: %.2f)",
                step, grad_norm, self._grad_norm_ema,
            )
            is_spike = True

        if not is_spike:
            return "ok"

        # --- Handle spike ---
        self._spike_count += 1
        self._rollback_count += 1

        if self._rollback_count > self.config.max_rollbacks:
            logger.error(
                "❌ Too many rollbacks (%d) — stopping training",
                self._rollback_count,
            )
            return "stop"

        logger.info(
            "🔄 Initiating rollback #%d (rolling back to step %d)",
            self._rollback_count, self._stable_step,
        )
        self._cooldown_until = step + self.config.cooldown_steps
        return "rollback"

    def rollback(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ) -> None:
        """Roll back model and optimizer to last stable state."""
        if self._stable_model_state is None:
            logger.warning("No stable state to rollback to!")
            return

        model.load_state_dict(self._stable_model_state)
        optimizer.load_state_dict(self._stable_optimizer_state)

        # Scale down learning rate
        scale = self.config.lr_scale_after_recovery
        for pg in optimizer.param_groups:
            pg["lr"] *= scale
        logger.info(
            "✅ Rolled back to step %d, LR scaled by %.2f",
            self._stable_step, scale,
        )

        # Reset EMA
        self._loss_ema = None
        self._grad_norm_ema = None

    def save_stable_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> None:
        """Save a stable checkpoint (for rollback)."""
        # Only save if recent losses are stable
        if len(self._loss_history) >= 10:
            recent = list(self._loss_history)[-10:]
            variance = sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent)
            mean_loss = sum(recent) / len(recent)
            cv = (variance ** 0.5) / max(mean_loss, 1e-8)  # coefficient of variation

            if cv > 0.5:
                logger.debug("Skipping stable checkpoint — loss too volatile (CV=%.3f)", cv)
                return

        self._save_state(model, optimizer)
        self._stable_step = step
        self._rollback_count = 0  # reset rollback counter on stable save
        logger.debug("Saved stable checkpoint at step %d", step)

    def _save_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Deep copy model and optimizer states."""
        self._stable_model_state = copy.deepcopy(model.state_dict())
        self._stable_optimizer_state = copy.deepcopy(optimizer.state_dict())

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_spikes": self._spike_count,
            "total_rollbacks": self._rollback_count,
            "current_loss_ema": self._loss_ema,
            "current_grad_norm_ema": self._grad_norm_ema,
            "stable_checkpoint_step": self._stable_step,
        }
