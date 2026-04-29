"""
titanbit.training.trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~
High-performance training loop for BitNet b1.58 models.

Engineered for maximum GPU utilisation on a single L20 (48GB):

    ┌─────────────────────────────────────────────────────────────┐
    │                    Training Pipeline                        │
    │                                                             │
    │  NVMe SSD ──mmap──▶ CPU ──pin_memory──▶ GPU                │
    │                                                             │
    │  ┌───────────┐   ┌──────────┐   ┌──────────┐              │
    │  │  Forward   │──▶│ Backward │──▶│ Optimize │              │
    │  │ (BF16 +   │   │ (STE     │   │ (AdamW)  │              │
    │  │  ternary)  │   │  grads)  │   │          │              │
    │  └───────────┘   └──────────┘   └──────────┘              │
    │       │                                │                    │
    │       ▼                                ▼                    │
    │  [StabilityMonitor]           [Checkpoint every N steps]   │
    │  [MFU Tracking]               [W&B Logging]                │
    └─────────────────────────────────────────────────────────────┘

Key features:
    - BF16 mixed precision (native on Ada Lovelace)
    - Gradient accumulation for effective batch size scaling
    - Cosine LR schedule with linear warmup
    - Periodic evaluation with validation perplexity
    - W&B integration for experiment tracking
    - Checkpoint save/resume for multi-day runs
    - MFU computation and throughput monitoring
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from titanbit.model.config import BitNetConfig
from titanbit.model.transformer import BitNetTransformer
from titanbit.training.data import DataConfig, create_data_loaders
from titanbit.training.stability import StabilityConfig, StabilityMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Training configuration."""

    # Optimisation
    learning_rate: float = 6e-4
    min_lr: float = 6e-5               # minimum LR (cosine decay target)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Schedule
    max_steps: int = 100_000
    warmup_steps: int = 2000

    # Batching
    batch_size: int = 8                # micro batch size (per GPU)
    gradient_accumulation_steps: int = 4  # effective batch = batch × accum

    # Precision
    dtype: str = "bfloat16"            # bfloat16 | float16 | float32
    compile_model: bool = True         # torch.compile for kernel fusion

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    eval_steps: int = 50
    save_interval: int = 2000

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    keep_last_n_checkpoints: int = 2       # keep only last N step checkpoints (+ best + final)
    auto_resume: bool = True               # auto-detect and resume from latest checkpoint
    resume_from: str = ""              # path to checkpoint to resume from

    # W&B
    wandb_project: str = "titanbit"
    wandb_run_name: str = ""
    use_wandb: bool = False

    # Gradient checkpointing (trades compute for memory)
    gradient_checkpointing: bool = False

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Stability
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainerConfig:
        data_cfg = DataConfig.from_dict(d.get("data", {}))
        stab_cfg = StabilityConfig.from_dict(d.get("stability", {}))
        
        # Cast numeric fields to prevent string vs float issues from YAML
        typed_dict = {}
        float_fields = ("learning_rate", "min_lr", "weight_decay", "beta1", "beta2", "max_grad_norm")
        int_fields = ("max_steps", "warmup_steps", "batch_size", "gradient_accumulation_steps", 
                      "log_interval", "eval_interval", "eval_steps", "save_interval", "keep_last_n_checkpoints")
        
        for k, v in d.items():
            if k in cls.__dataclass_fields__ and k not in ("data", "stability"):
                if k in float_fields:
                    typed_dict[k] = float(v) if v is not None else v
                elif k in int_fields:
                    typed_dict[k] = int(v) if v is not None else v
                else:
                    typed_dict[k] = v
                    
        return cls(**typed_dict, data=data_cfg, stability=stab_cfg)

    @classmethod
    def from_yaml(cls, path: str) -> TrainerConfig:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw.get("training", {}))

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training engine for BitNet b1.58.

    Usage
    -----
    >>> model_cfg = BitNetConfig(hidden_size=2048, num_layers=24, num_heads=32)
    >>> train_cfg = TrainerConfig.from_yaml("configs/default.yaml")
    >>> trainer = Trainer(model_cfg, train_cfg)
    >>> trainer.train()
    """

    def __init__(
        self,
        model_config: BitNetConfig,
        trainer_config: TrainerConfig,
        model: Optional[BitNetTransformer] = None,
    ) -> None:
        self.model_config = model_config
        self.config = trainer_config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[trainer_config.dtype]

        logger.info("Device: %s | Dtype: %s", self.device, self.dtype)

        # Model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = BitNetTransformer(model_config).to(self.device)

        if trainer_config.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        # Compile model for kernel fusion (PyTorch 2.0+)
        if trainer_config.compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile(mode='max-autotune')...")
            self.model = torch.compile(self.model, mode="max-autotune")

        # Optimizer
        self.optimizer = self._create_optimizer()

        # LR scheduler (cosine with warmup)
        self.scheduler = None  # managed manually in the training loop

        # Stability monitor
        self.stability = StabilityMonitor(
            trainer_config.stability,
            self.model,
            self.optimizer,
        )

        # Data
        self.train_loader, self.val_loader = create_data_loaders(trainer_config.data)

        # AMP
        self.scaler = GradScaler() if self.dtype == torch.float16 else None
        self.amp_ctx = torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        )

        # Tracking
        self.global_step = 0
        self.tokens_processed = 0
        self.best_val_loss = float("inf")
        self._train_start_time = 0.0

        # W&B
        self._wandb_run = None
        if trainer_config.use_wandb:
            self._init_wandb()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with weight decay only on 2D parameters.

        Following the standard practice from GPT-2/3: we don't apply
        weight decay to biases, LayerNorm/RMSNorm weights, or embeddings.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't decay 1D params (norms, biases) or embeddings
            if param.ndim < 2 or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            fused=self.device.type == "cuda",  # fused AdamW for CUDA
        )

        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in no_decay_params)
        logger.info(
            "Optimizer: AdamW | decay params: %s | no-decay params: %s",
            f"{n_decay:,}", f"{n_nodecay:,}",
        )
        return optimizer

    def _get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        cfg = self.config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step / max(1, cfg.warmup_steps))

        # Cosine decay
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(1.0, progress)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * coeff

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ----- Training -----

    def train(self) -> dict[str, Any]:
        """
        Main training loop.

        Supports:
            - Auto-resume from latest checkpoint in checkpoint_dir
            - Explicit resume via resume_from path
            - Checkpoint rotation (keep last N + best + final)
            - Data exhaustion detection (stops if data runs out)
            - Epoch tracking to prevent overfitting

        Returns a dict with final training statistics.
        """
        cfg = self.config
        model = self.model
        optimizer = self.optimizer

        # Resume logic: explicit > auto-detect > fresh start
        if cfg.resume_from and os.path.exists(cfg.resume_from):
            self._load_checkpoint(cfg.resume_from)
        elif cfg.auto_resume:
            latest = self._find_latest_checkpoint()
            if latest:
                self._load_checkpoint(latest)

        logger.info("=" * 70)
        logger.info("TitanBit Training")
        logger.info("  Model:          %s params", self.model_config.num_params_str)
        logger.info("  Effective batch: %d", cfg.effective_batch_size)
        logger.info("  Max steps:       %d", cfg.max_steps)
        logger.info("  Warmup steps:    %d", cfg.warmup_steps)
        logger.info("  Learning rate:   %.2e -> %.2e", cfg.learning_rate, cfg.min_lr)
        logger.info("  Precision:       %s", cfg.dtype)
        logger.info("  Checkpoint rotation: keep last %d", cfg.keep_last_n_checkpoints)
        if self.global_step > 0:
            logger.info("  Resumed from:    step %d (%s tokens)",
                        self.global_step, f"{self.tokens_processed:,}")
        logger.info("=" * 70)

        self._train_start_time = time.monotonic()
        train_iter = iter(self.train_loader)
        data_exhausted = False

        while self.global_step < cfg.max_steps:
            # Set LR for this step
            lr = self._get_lr(self.global_step)
            self._set_lr(lr)

            model.train()

            # Gradient accumulation
            micro_losses = []
            for micro_step in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # Data exhausted — this is the anti-overfit signal
                    logger.warning(
                        "Data exhausted at step %d (%s tokens). "
                        "All data has been consumed — stopping to prevent overfitting.",
                        self.global_step, f"{self.tokens_processed:,}",
                    )
                    data_exhausted = True
                    break

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                with self.amp_ctx:
                    logits, loss = model(input_ids, labels=labels)
                    loss = loss / cfg.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_losses.append(loss.item() * cfg.gradient_accumulation_steps)

            if data_exhausted:
                break

            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
            grad_norm = self.stability.clip_gradients(model)

            # Stability check
            avg_loss = sum(micro_losses) / len(micro_losses)
            action = self.stability.check(self.global_step, avg_loss, grad_norm)

            if action == "rollback":
                self.stability.rollback(model, optimizer)
                optimizer.zero_grad(set_to_none=True)
                continue
            elif action == "stop":
                logger.error("Training stopped due to instability")
                break

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Update tracking
            self.global_step += 1
            batch_tokens = cfg.effective_batch_size * self.model_config.max_seq_length
            self.tokens_processed += batch_tokens

            # Logging
            if self.global_step % cfg.log_interval == 0:
                self._log_step(avg_loss, grad_norm, lr)

            # Evaluation
            if self.global_step % cfg.eval_interval == 0 and self.val_loader is not None:
                val_loss = self._evaluate()
                self._log_eval(val_loss)

            # Save checkpoint (with rotation)
            if self.global_step % cfg.save_interval == 0:
                self._save_checkpoint()
                self._rotate_checkpoints()

            # Save stable state for rollback
            if self.global_step % cfg.stability.stable_checkpoint_interval == 0:
                self.stability.save_stable_state(model, optimizer, self.global_step)

        # Final save
        self._save_checkpoint(final=True)
        elapsed = time.monotonic() - self._train_start_time

        stats = {
            "total_steps": self.global_step,
            "tokens_processed": self.tokens_processed,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self.tokens_processed / elapsed if elapsed > 0 else 0,
            "best_val_loss": self.best_val_loss,
            "data_exhausted": data_exhausted,
            "stability": self.stability.stats,
        }

        logger.info("=" * 70)
        logger.info("Training complete!")
        logger.info("  Steps: %d | Tokens: %s", self.global_step, f"{self.tokens_processed:,}")
        logger.info("  Time: %.1f hours", elapsed / 3600)
        logger.info("  Throughput: %.0f tokens/sec", stats["tokens_per_second"])
        if data_exhausted:
            logger.info("  Data exhausted: YES (single epoch completed, no overfit risk)")
        logger.info("=" * 70)

        return stats

    # ----- Evaluation -----

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Run evaluation on validation set."""
        model = self.model
        model.eval()

        total_loss = 0.0
        count = 0

        for i, batch in enumerate(self.val_loader):
            if i >= self.config.eval_steps:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with self.amp_ctx:
                _, loss = model(input_ids, labels=labels)

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(best=True)

        return avg_loss

    # ----- Logging -----

    def _log_step(self, loss: float, grad_norm: float, lr: float) -> None:
        elapsed = time.monotonic() - self._train_start_time
        tokens_per_sec = self.tokens_processed / max(elapsed, 1e-6)

        # MFU estimate
        dt = elapsed / max(self.global_step, 1)
        if hasattr(self.model, "estimate_mfu"):
            mfu = self.model.estimate_mfu(
                self.config.batch_size,
                self.model_config.max_seq_length,
                dt,
            )
        elif hasattr(self.model, "_orig_mod"):
            # torch.compile wraps the model
            mfu = self.model._orig_mod.estimate_mfu(
                self.config.batch_size,
                self.model_config.max_seq_length,
                dt,
            )
        else:
            mfu = 0.0

        # VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            vram_str = f"{vram_used:.1f}/{vram_total:.1f}GB"
        else:
            vram_str = "N/A"

        logger.info(
            "step=%d | loss=%.4f | grad=%.3f | lr=%.2e | "
            "tok/s=%.0f | MFU=%.1f%% | VRAM=%s",
            self.global_step, loss, grad_norm, lr,
            tokens_per_sec, mfu * 100, vram_str,
        )

        if self._wandb_run is not None:
            import wandb
            wandb.log({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_sec,
                "train/mfu": mfu,
                "train/step": self.global_step,
                "train/tokens": self.tokens_processed,
            }, step=self.global_step)

    def _log_eval(self, val_loss: float) -> None:
        ppl = math.exp(min(val_loss, 20))
        logger.info(
            "📊 EVAL step=%d | val_loss=%.4f | val_ppl=%.2f | best_loss=%.4f",
            self.global_step, val_loss, ppl, self.best_val_loss,
        )

        if self._wandb_run is not None:
            import wandb
            wandb.log({
                "eval/loss": val_loss,
                "eval/perplexity": ppl,
                "eval/best_loss": self.best_val_loss,
            }, step=self.global_step)

    # ----- Checkpointing -----

    def _save_checkpoint(self, best: bool = False, final: bool = False) -> None:
        """Save training checkpoint."""
        ckpt_dir = self.config.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # Determine filename
        if final:
            name = "checkpoint_final.pt"
        elif best:
            name = "checkpoint_best.pt"
        else:
            name = f"checkpoint_step_{self.global_step:07d}.pt"

        path = os.path.join(ckpt_dir, name)

        # Get the base model (unwrap torch.compile if needed)
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config.to_dict(),
            "trainer_config_snapshot": {
                "learning_rate": self.config.learning_rate,
                "max_steps": self.config.max_steps,
                "dtype": self.config.dtype,
            },
            "global_step": self.global_step,
            "tokens_processed": self.tokens_processed,
            "best_val_loss": self.best_val_loss,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s (step %d)", path, self.global_step)

    def _rotate_checkpoints(self) -> None:
        """
        Keep only the last N step checkpoints + best + final.

        This prevents disk from filling up during multi-day runs.
        The 'best' and 'final' checkpoints are NEVER deleted.
        """
        keep_n = self.config.keep_last_n_checkpoints
        if keep_n <= 0:
            return  # rotation disabled

        ckpt_dir = self.config.checkpoint_dir
        pattern = os.path.join(ckpt_dir, "checkpoint_step_*.pt")
        step_files = sorted(glob.glob(pattern))

        # Keep only the most recent N
        if len(step_files) > keep_n:
            to_delete = step_files[:-keep_n]
            for f in to_delete:
                try:
                    os.remove(f)
                    logger.debug("Rotated old checkpoint: %s", f)
                except OSError as e:
                    logger.warning("Failed to delete checkpoint %s: %s", f, e)

            remaining = len(step_files) - len(to_delete)
            logger.info(
                "Checkpoint rotation: deleted %d old, keeping %d recent",
                len(to_delete), remaining,
            )

    def _find_latest_checkpoint(self) -> Optional[str]:
        """
        Auto-detect the latest step checkpoint in checkpoint_dir.

        Searches for checkpoint_step_NNNNNNN.pt files and returns
        the one with the highest step number.
        """
        ckpt_dir = self.config.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return None

        pattern = os.path.join(ckpt_dir, "checkpoint_step_*.pt")
        step_files = sorted(glob.glob(pattern))

        if step_files:
            latest = step_files[-1]
            logger.info("Auto-resume: found latest checkpoint %s", latest)
            return latest

        # Also check for final checkpoint
        final_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
        if os.path.exists(final_path):
            logger.info("Auto-resume: found final checkpoint %s", final_path)
            return final_path

        return None

    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint and resume."""
        logger.info("Loading checkpoint from %s...", path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Get the base model (unwrap torch.compile if needed)
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.tokens_processed = checkpoint.get("tokens_processed", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            "Resumed from step %d (%s tokens processed)",
            self.global_step, f"{self.tokens_processed:,}",
        )

    # ----- W&B -----

    def _init_wandb(self) -> None:
        try:
            import wandb
            run_name = self.config.wandb_run_name or (
                f"bitnet-{self.model_config.num_params_str}-"
                f"lr{self.config.learning_rate}"
            )
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "model": self.model_config.to_dict(),
                    "training": {
                        "lr": self.config.learning_rate,
                        "batch_size": self.config.effective_batch_size,
                        "max_steps": self.config.max_steps,
                        "dtype": self.config.dtype,
                    },
                },
            )
            logger.info("W&B initialised: %s/%s", self.config.wandb_project, run_name)
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")
        except Exception as e:
            logger.warning("W&B init failed: %s", e)
