"""
Tests for the BitLinear layer — the core of BitNet b1.58.

Tests verify:
    1. Weight quantisation produces only ternary values {-1, 0, 1}
    2. Activation quantisation stays within int8 range [-128, 127]
    3. STE gradients flow through correctly
    4. BitLinear output shape matches nn.Linear
    5. RMSNorm output has unit RMS
    6. Inference mode produces identical results with pre-quantised weights
    7. Weight packing/unpacking round-trips correctly
"""

import pytest
import torch
import torch.nn as nn

from titanbit.model.bitlinear import (
    BitLinear,
    RMSNorm,
    quantise_weights,
    quantise_activations,
)


class TestWeightQuantisation:
    """Test ternary weight quantisation."""

    def test_output_is_ternary(self):
        """Quantised weights should only contain {-1, 0, 1}."""
        w = torch.randn(256, 128)
        w_q, gamma = quantise_weights(w)
        unique = torch.unique(w_q)
        for v in unique:
            assert v.item() in {-1.0, 0.0, 1.0}, f"Unexpected value: {v}"

    def test_scale_is_positive(self):
        """The scaling factor γ should be positive."""
        w = torch.randn(64, 64)
        _, gamma = quantise_weights(w)
        assert gamma.item() > 0

    def test_zero_weights(self):
        """Zero weights should quantise to zero."""
        w = torch.zeros(32, 32)
        w_q, _ = quantise_weights(w)
        assert (w_q == 0).all()

    def test_gradient_flows(self):
        """STE should allow gradients to flow through quantisation."""
        w = torch.randn(64, 64, requires_grad=True)
        w_q, gamma = quantise_weights(w)
        loss = w_q.sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.shape == w.shape
        # STE: gradient should be all ones (since loss = sum(w_q))
        assert torch.allclose(w.grad, torch.ones_like(w.grad))

    def test_deterministic(self):
        """Quantisation should be deterministic."""
        w = torch.randn(64, 64)
        w_q1, g1 = quantise_weights(w)
        w_q2, g2 = quantise_weights(w)
        assert torch.equal(w_q1, w_q2)
        assert torch.equal(g1, g2)


class TestActivationQuantisation:
    """Test 8-bit activation quantisation."""

    def test_output_range(self):
        """Quantised activations should be in [-128, 127]."""
        x = torch.randn(4, 128, 256)
        x_q, eta = quantise_activations(x)
        assert x_q.min() >= -128
        assert x_q.max() <= 127

    def test_scale_is_positive(self):
        """The scaling factor η should be positive."""
        x = torch.randn(4, 128, 256)
        _, eta = quantise_activations(x)
        assert (eta > 0).all()

    def test_gradient_flows(self):
        """STE should allow gradients to flow through."""
        x = torch.randn(4, 128, 256, requires_grad=True)
        x_q, eta = quantise_activations(x)
        loss = x_q.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_batch_independence(self):
        """Each token should be quantised independently."""
        x = torch.randn(2, 4, 64)
        x_q, eta = quantise_activations(x)
        # η should have shape (2, 4, 1) — per-token scaling
        assert eta.shape == (2, 4, 1)


class TestRMSNorm:
    """Test RMSNorm layer."""

    def test_output_shape(self):
        """Output shape should match input."""
        norm = RMSNorm(256)
        x = torch.randn(4, 128, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_rms(self):
        """After normalisation, RMS should be approximately 1."""
        norm = RMSNorm(256)
        x = torch.randn(4, 128, 256)
        out = norm(x)
        rms = out.float().pow(2).mean(dim=-1).sqrt()
        # Should be close to 1 (the weight is initialised to ones)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_dtype_preservation(self):
        """Output dtype should match input dtype."""
        norm = RMSNorm(128)
        x_bf16 = torch.randn(2, 64, 128, dtype=torch.bfloat16)
        out = norm(x_bf16)
        assert out.dtype == torch.bfloat16


class TestBitLinear:
    """Test the complete BitLinear layer."""

    def test_output_shape(self):
        """Output shape should be (batch, seq, out_features)."""
        layer = BitLinear(256, 512)
        x = torch.randn(2, 128, 256)
        out = layer(x)
        assert out.shape == (2, 128, 512)

    def test_no_bias(self):
        """BitLinear should not have a bias parameter."""
        layer = BitLinear(128, 256)
        assert layer.bias is None

    def test_gradient_updates_shadow_weights(self):
        """Gradients should update the full-precision shadow weights."""
        layer = BitLinear(64, 128)
        x = torch.randn(1, 16, 64)

        # Forward + backward
        out = layer(x)
        loss = out.sum()
        loss.backward()

        # Shadow weight should have gradients
        assert layer.weight.grad is not None
        # Gradient values should not all be the same
        assert not torch.all(layer.weight.grad == 0)

    def test_inference_mode(self):
        """Inference mode should produce similar results to training mode."""
        layer = BitLinear(128, 256)
        x = torch.randn(1, 32, 128)

        # Training forward
        layer.eval()
        with torch.no_grad():
            out_train = layer(x)

        # Switch to inference mode
        layer.prepare_for_inference()
        with torch.no_grad():
            out_infer = layer(x)

        # Results should be identical (both use the same quantised weights)
        assert torch.allclose(out_train, out_infer, atol=1e-5)

    def test_matches_linear_shape(self):
        """BitLinear should have the same interface as nn.Linear."""
        bit_layer = BitLinear(256, 512)
        lin_layer = nn.Linear(256, 512, bias=False)

        x = torch.randn(2, 64, 256)
        bit_out = bit_layer(x)
        lin_out = lin_layer(x)

        assert bit_out.shape == lin_out.shape

    def test_bf16_compatible(self):
        """BitLinear should work with BF16 inputs."""
        layer = BitLinear(128, 256)
        x = torch.randn(2, 32, 128, dtype=torch.bfloat16)
        # Need to cast layer weights to bfloat16
        layer = layer.to(torch.bfloat16)
        out = layer(x)
        assert out.shape == (2, 32, 256)


class TestWeightPacking:
    """Test ternary weight packing/unpacking."""

    def test_roundtrip(self):
        """Packing then unpacking should recover original weights."""
        from titanbit.model.kernels import pack_ternary_weights, unpack_ternary_weights

        w = torch.randint(-1, 2, (64, 128)).float()
        packed = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, in_features=128)

        assert torch.equal(w, unpacked)

    def test_compression_ratio(self):
        """Packed representation should be 8× smaller than float32."""
        from titanbit.model.kernels import pack_ternary_weights

        w = torch.randint(-1, 2, (256, 256)).float()
        packed = pack_ternary_weights(w)

        # Original: 256 × 256 × 4 bytes = 262144 bytes
        # Packed: 256 × 16 × 4 bytes = 16384 bytes → 16× compression
        assert packed.shape == (256, 16)  # 256 / 16 = 16

    def test_all_values(self):
        """Packing should handle all three ternary values."""
        from titanbit.model.kernels import pack_ternary_weights, unpack_ternary_weights

        # Create a weight with all three values
        w = torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, 0]]).float()
        packed = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, in_features=16)
        assert torch.equal(w, unpacked)
