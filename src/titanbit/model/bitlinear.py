"""
titanbit.model.bitlinear
~~~~~~~~~~~~~~~~~~~~~~~~~
Core BitLinear layer implementing BitNet b1.58 quantisation.

This is the fundamental building block that replaces nn.Linear throughout
the transformer.  During training, full-precision "shadow weights" are
maintained for gradient accumulation, while the forward pass operates
with ternary {-1, 0, 1} weights and 8-bit quantised activations.

Architecture (from Ma et al., 2024 — "The Era of 1-bit LLMs"):
    ┌──────────────┐
    │   Input x    │
    │   (BF16)     │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   RMSNorm    │  ← SubLN: stabilises input distribution
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  AbsMean     │  ← Quantise activations to 8-bit
    │  Quant (8b)  │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  W_ternary   │  ← Weights quantised to {-1, 0, 1}
    │  MatMul      │     via round-clip of normalised weights
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Rescale     │  ← Multiply by β (weight scale) × γ (act scale)
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Output     │
    │   (BF16)     │
    └──────────────┘

Key engineering decisions:
    1. STE (Straight-Through Estimator) — Gradients flow through the
       quantisation step unchanged.  This is standard practice for QAT.
    2. Per-tensor AbsMean scaling — Cheaper than per-channel and sufficient
       for the ternary regime (validated in the BitNet paper).
    3. No bias — Removed to maximise the benefit of ternary compression.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quantisation primitives
# ---------------------------------------------------------------------------

class WeightQuantiser(torch.autograd.Function):
    """
    Quantise weights to ternary {-1, 0, 1} with Straight-Through Estimator.

    Forward:
        1. Compute per-tensor mean absolute value: γ = mean(|W|)
        2. Normalise: W_norm = W / (γ + ε)
        3. Round to nearest integer and clip to [-1, 1]
        4. Scale back: W_q = round_clip(W_norm) — stored as the ternary weight
        5. Return (W_q, γ) where γ is the scaling factor

    Backward:
        STE: ∂L/∂W = ∂L/∂W_q  (gradient passes through unchanged)

    This is the core of BitNet b1.58.  The ternary constraint means each
    weight is encoded in log2(3) ≈ 1.58 bits.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-tensor absolute mean
        gamma = weight.abs().mean().clamp(min=1e-5)
        # Normalise, round, clip
        w_norm = weight / gamma
        w_q = w_norm.round().clamp(-1, 1)
        return w_q, gamma

    @staticmethod
    def backward(ctx, grad_wq: torch.Tensor, grad_gamma: torch.Tensor) -> torch.Tensor:
        # STE: pass gradient through unchanged
        return grad_wq


class ActivationQuantiser(torch.autograd.Function):
    """
    Quantise activations to 8-bit using AbsMean scaling.

    Forward:
        1. Compute per-token absolute max: η = max(|x|, dim=-1)
        2. Scale to [-128, 127] range:  x_q = clamp(round(x × 127 / η), -128, 127)
        3. Return (x_q, η)

    Backward:
        STE: gradient passes through unchanged.

    Using AbsMean (per-token) rather than per-channel gives better
    throughput on modern GPUs because it avoids scatter/gather ops.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-token max absolute value
        eta = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        # Scale to int8 range
        x_scaled = x * (127.0 / eta)
        x_q = x_scaled.round().clamp(-128, 127)
        return x_q, eta

    @staticmethod
    def backward(ctx, grad_xq: torch.Tensor, grad_eta: torch.Tensor) -> torch.Tensor:
        # STE: pass gradient through unchanged
        return grad_xq


def quantise_weights(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional interface to weight quantisation."""
    return WeightQuantiser.apply(weight)


def quantise_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional interface to activation quantisation."""
    return ActivationQuantiser.apply(x)


# ---------------------------------------------------------------------------
# RMSNorm (for SubLN inside BitLinear)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Simpler and faster than LayerNorm: no mean subtraction, no bias.
    Standard in modern LLMs (LLaMA, Mistral, etc.).

        RMSNorm(x) = x / RMS(x) × γ
        RMS(x) = sqrt(mean(x²) + ε)
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for numerical stability, then cast back
        x_fp32 = x.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x_fp32 / rms
        return (x_norm * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# BitLinear layer
# ---------------------------------------------------------------------------

class BitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with BitNet b1.58 quantisation.

    During training:
        - Maintains full-precision (BF16/FP32) shadow weights
        - Forward pass: quantise weights → ternary, quantise activations → 8-bit
        - Backward pass: STE sends gradients to shadow weights
        - Optimizer updates shadow weights in full precision

    During inference:
        - Weights are pre-quantised to ternary (stored as int8)
        - Activations are quantised on-the-fly
        - MatMul uses integer arithmetic (via Triton kernel)

    Parameters
    ----------
    in_features  : input dimension
    out_features : output dimension
    bias         : always False for BitNet (ignored)

    Example
    -------
    >>> layer = BitLinear(768, 3072)
    >>> x = torch.randn(2, 128, 768, dtype=torch.bfloat16)
    >>> out = layer(x)
    >>> out.shape
    torch.Size([2, 128, 3072])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,     # always False for BitNet
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Shadow weight — full precision, used for gradient accumulation
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # SubLN: RMSNorm applied to input before quantisation
        self.input_norm = RMSNorm(in_features)

        # No bias in BitNet
        self.register_parameter("bias", None)

        # Initialisation: scaled normal (Kaiming-inspired)
        self._reset_parameters()

        # Flag for inference mode (pre-quantised weights)
        self._inference_mode = False
        self._w_quantised: Optional[torch.Tensor] = None
        self._w_scale: Optional[torch.Tensor] = None

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantised weights and activations.

        Computation graph:
            x → RMSNorm → AbsMean_quant(8-bit) → matmul(ternary_W) → rescale → out
        """
        if self._inference_mode:
            return self._forward_inference(x)

        # ---- Training forward ----

        # 1. SubLN: normalise input
        x_norm = self.input_norm(x)

        # 2. Quantise activations to 8-bit
        x_q, x_scale = quantise_activations(x_norm)

        # 3. Quantise weights to ternary
        w_q, w_scale = quantise_weights(self.weight)

        # 4. Integer-like matmul (still in float for autograd compatibility)
        # In training we keep everything in the same dtype for gradient flow
        out = F.linear(x_q.to(x.dtype), w_q.to(x.dtype))

        # 5. Rescale: undo the quantisation scaling
        # out_real = (x_q / 127 × x_scale) @ (w_q × w_scale)^T
        #          = (x_q @ w_q^T) × (x_scale × w_scale / 127)
        out = out * (x_scale * w_scale / 127.0)

        return out

    def _forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Optimised inference with pre-quantised weights."""
        x_norm = self.input_norm(x)
        x_q, x_scale = quantise_activations(x_norm)

        # Use pre-quantised ternary weights (int8 storage)
        out = F.linear(x_q.to(x.dtype), self._w_quantised.to(x.dtype))
        out = out * (x_scale * self._w_scale / 127.0)
        return out

    def prepare_for_inference(self) -> None:
        """Pre-quantise weights for inference (saves per-forward cost)."""
        with torch.no_grad():
            gamma = self.weight.abs().mean().clamp(min=1e-5)
            w_q = (self.weight / gamma).round().clamp(-1, 1).to(torch.int8)
            self._w_quantised = w_q
            self._w_scale = gamma
            self._inference_mode = True

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits=1.58 (ternary), act_bits=8"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_ternary_params(model: nn.Module) -> dict[str, int]:
    """Count parameters that are ternary-quantised vs full-precision."""
    ternary = 0
    full = 0
    for name, param in model.named_parameters():
        if isinstance(getattr_nested(model, name.rsplit(".", 1)[0]), BitLinear):
            if name.endswith(".weight"):
                ternary += param.numel()
                continue
        full += param.numel()
    return {"ternary": ternary, "full_precision": full, "total": ternary + full}


def getattr_nested(obj: object, name: str) -> object:
    """Get a nested attribute by dot-separated name."""
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj
