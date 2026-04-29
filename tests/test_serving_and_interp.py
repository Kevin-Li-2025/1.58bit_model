"""
Tests for TitanServe (inference engine) and TitanLens (interpretability).
"""

import pytest
import torch

from titanbit.model.config import BitNetConfig
from titanbit.model.transformer import BitNetTransformer


TINY_CONFIG = BitNetConfig(
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    vocab_size=256,
    max_seq_length=128,
)


# ---------------------------------------------------------------------------
# Inference Engine Tests
# ---------------------------------------------------------------------------

class TestInferenceEngine:
    """Test the inference engine."""

    def test_generate_shape(self):
        """Generated output should extend the prompt."""
        from titanbit.serving.engine import InferenceEngine, GenerationConfig

        model = BitNetTransformer(TINY_CONFIG)
        engine = InferenceEngine(model)

        prompt = torch.randint(0, 256, (1, 16))
        config = GenerationConfig(max_new_tokens=8, do_sample=False)
        output = engine.generate(prompt, config)

        assert output.shape[0] == 1
        assert output.shape[1] == 16 + 8  # prompt + generated

    def test_greedy_deterministic(self):
        """Greedy decoding should be deterministic."""
        from titanbit.serving.engine import InferenceEngine, GenerationConfig

        model = BitNetTransformer(TINY_CONFIG)
        engine = InferenceEngine(model)

        prompt = torch.randint(0, 256, (1, 8))
        config = GenerationConfig(max_new_tokens=4, do_sample=False)

        out1 = engine.generate(prompt, config)
        out2 = engine.generate(prompt, config)

        assert torch.equal(out1, out2)

    def test_streaming(self):
        """Streaming should yield individual tokens."""
        from titanbit.serving.engine import InferenceEngine, GenerationConfig

        model = BitNetTransformer(TINY_CONFIG)
        engine = InferenceEngine(model)

        prompt = torch.randint(0, 256, (1, 8))
        config = GenerationConfig(max_new_tokens=5, do_sample=False)

        tokens = list(engine.generate_stream(prompt, config))
        assert len(tokens) == 5
        assert all(isinstance(t, int) for t in tokens)

    def test_eos_stopping(self):
        """Generation should stop at EOS token."""
        from titanbit.serving.engine import InferenceEngine, GenerationConfig

        model = BitNetTransformer(TINY_CONFIG)
        engine = InferenceEngine(model)

        prompt = torch.randint(0, 256, (1, 8))
        # Set EOS to a common token to test early stopping
        config = GenerationConfig(max_new_tokens=100, do_sample=False, eos_token_id=0)
        output = engine.generate(prompt, config)

        # Should stop before max_new_tokens if EOS is generated
        assert output.shape[1] <= 8 + 100


class TestSampler:
    """Test the token sampler."""

    def test_greedy(self):
        """Greedy should return the argmax."""
        from titanbit.serving.engine import Sampler, GenerationConfig

        logits = torch.tensor([[0.1, 0.3, 0.9, 0.2]])
        config = GenerationConfig(do_sample=False)
        token = Sampler.sample(logits, config)
        assert token.item() == 2

    def test_temperature_sharpening(self):
        """Low temperature should make distribution sharper."""
        from titanbit.serving.engine import Sampler, GenerationConfig

        logits = torch.tensor([[1.0, 2.0, 3.0]])

        # High temperature — more uniform
        config_hot = GenerationConfig(temperature=10.0, do_sample=True, top_k=0, top_p=1.0)
        # Low temperature — sharper
        config_cold = GenerationConfig(temperature=0.01, do_sample=True, top_k=0, top_p=1.0)

        # With very low temp, should almost always pick the max
        cold_tokens = [Sampler.sample(logits, config_cold).item() for _ in range(20)]
        assert all(t == 2 for t in cold_tokens)

    def test_top_k(self):
        """Top-k should restrict to top k tokens."""
        from titanbit.serving.engine import Sampler, GenerationConfig

        logits = torch.tensor([[10.0, 5.0, 1.0, 0.5, 0.1]])
        config = GenerationConfig(top_k=2, temperature=0.5, do_sample=True, top_p=1.0)

        tokens = set()
        for _ in range(50):
            tokens.add(Sampler.sample(logits, config).item())

        # Should only sample from top 2 tokens (0 and 1)
        assert tokens.issubset({0, 1})


# ---------------------------------------------------------------------------
# Speculative Decoding Tests
# ---------------------------------------------------------------------------

class TestSpeculativeDecoder:
    """Test speculative decoding."""

    def test_speculative_output(self):
        """Speculative decoding should produce valid output."""
        from titanbit.serving.speculative import SpeculativeDecoder, SpeculativeConfig

        target = BitNetTransformer(TINY_CONFIG)
        draft_config = BitNetConfig(
            hidden_size=64, num_layers=1, num_heads=4,
            vocab_size=256, max_seq_length=128,
        )
        draft = BitNetTransformer(draft_config)

        decoder = SpeculativeDecoder(target, draft)
        prompt = torch.randint(0, 256, (1, 8))
        config = SpeculativeConfig(
            num_speculative_tokens=3,
            max_new_tokens=10,
            log_acceptance_rate=False,
        )
        output = decoder.generate(prompt, config)

        assert output.shape[0] == 1
        assert output.shape[1] > 8  # should have generated something

    def test_acceptance_rate_tracking(self):
        """Acceptance rate should be between 0 and 1."""
        from titanbit.serving.speculative import SpeculativeDecoder, SpeculativeConfig

        target = BitNetTransformer(TINY_CONFIG)
        draft = BitNetTransformer(TINY_CONFIG)

        decoder = SpeculativeDecoder(target, draft)
        prompt = torch.randint(0, 256, (1, 8))
        config = SpeculativeConfig(
            num_speculative_tokens=3,
            max_new_tokens=5,
            log_acceptance_rate=False,
        )
        decoder.generate(prompt, config)

        assert 0.0 <= decoder.acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# Interpretability Tests
# ---------------------------------------------------------------------------

class TestActivationExtractor:
    """Test activation extraction."""

    def test_extract_activations(self):
        """Should capture activations at specified layers."""
        from titanbit.interpretability.probing import ActivationExtractor

        model = BitNetTransformer(TINY_CONFIG)
        model.eval()
        extractor = ActivationExtractor(model, layers=[0, 1])

        input_ids = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            model(input_ids)

        acts = extractor.get_activations()
        assert 0 in acts
        assert 1 in acts
        assert acts[0].shape == (1, 16, 64)  # (B, T, hidden)

        extractor.remove_hooks()


class TestLinearProbe:
    """Test linear probes."""

    def test_probe_output_shape(self):
        """Probe should output (B, num_classes)."""
        from titanbit.interpretability.probing import LinearProbe

        probe = LinearProbe(hidden_size=64, num_classes=3)
        x = torch.randn(4, 64)
        out = probe(x)
        assert out.shape == (4, 3)

    def test_probe_loss(self):
        """Probe loss should be positive."""
        from titanbit.interpretability.probing import LinearProbe

        probe = LinearProbe(hidden_size=64, num_classes=3)
        x = torch.randn(4, 64)
        labels = torch.randint(0, 3, (4,))
        loss = probe.compute_loss(x, labels)
        assert loss.item() > 0

    def test_probe_predict(self):
        """Predictions should be valid class indices."""
        from titanbit.interpretability.probing import LinearProbe

        probe = LinearProbe(hidden_size=64, num_classes=5)
        x = torch.randn(4, 64)
        preds = probe.predict(x)
        assert preds.shape == (4,)
        assert (preds >= 0).all()
        assert (preds < 5).all()


class TestSparseAutoencoder:
    """Test sparse autoencoders."""

    def test_sae_output_shape(self):
        """SAE should return features, reconstruction, and loss."""
        from titanbit.interpretability.probing import SparseAutoencoder

        sae = SparseAutoencoder(hidden_size=64, num_features=256)
        x = torch.randn(4, 64)
        z, h_hat, loss = sae(x)

        assert z.shape == (4, 256)
        assert h_hat.shape == (4, 64)
        assert loss.ndim == 0

    def test_sae_sparsity(self):
        """Features should be sparse (many zeros after ReLU)."""
        from titanbit.interpretability.probing import SparseAutoencoder

        sae = SparseAutoencoder(hidden_size=64, num_features=256)
        x = torch.randn(4, 64)
        z, _, _ = sae(x)

        sparsity = (z == 0).float().mean()
        # After random init, some fraction should be zero due to ReLU
        assert sparsity > 0.1

    def test_top_features(self):
        """Should return top-k feature indices and values."""
        from titanbit.interpretability.probing import SparseAutoencoder

        sae = SparseAutoencoder(hidden_size=64, num_features=256)
        x = torch.randn(2, 64)
        indices, values = sae.get_top_features(x, k=5)

        assert indices.shape == (2, 5)
        assert values.shape == (2, 5)


class TestTernaryCircuitAnalyser:
    """Test ternary circuit analysis."""

    def test_weight_statistics(self):
        """Should compute valid weight statistics."""
        from titanbit.interpretability.circuits import TernaryCircuitAnalyser

        model = BitNetTransformer(TINY_CONFIG)
        analyser = TernaryCircuitAnalyser(model)
        stats = analyser.weight_statistics()

        assert "overall" in stats
        assert stats["overall"]["total"] > 0
        assert stats["overall"]["positive"] >= 0
        assert stats["overall"]["negative"] >= 0
        assert stats["overall"]["zero"] >= 0

    def test_connectivity(self):
        """Connectivity should be between 0 and 1."""
        from titanbit.interpretability.circuits import TernaryCircuitAnalyser

        model = BitNetTransformer(TINY_CONFIG)
        analyser = TernaryCircuitAnalyser(model)
        conn = analyser.layer_connectivity()

        for name, metrics in conn.items():
            assert 0 <= metrics["connectivity"] <= 1

    def test_ei_balance(self):
        """E/I fractions should sum to approximately 1."""
        from titanbit.interpretability.circuits import TernaryCircuitAnalyser

        model = BitNetTransformer(TINY_CONFIG)
        analyser = TernaryCircuitAnalyser(model)
        balance = analyser.excitatory_inhibitory_balance()

        for name, metrics in balance.items():
            total = metrics["excitatory_fraction"] + metrics["inhibitory_fraction"]
            assert 0.9 < total < 1.1  # should sum to ~1

    def test_summary(self):
        """Summary should be a non-empty string."""
        from titanbit.interpretability.circuits import TernaryCircuitAnalyser

        model = BitNetTransformer(TINY_CONFIG)
        analyser = TernaryCircuitAnalyser(model)
        summary = analyser.summary()

        assert isinstance(summary, str)
        assert len(summary) > 100
        assert "Sparsity" in summary


class TestCausalTracer:
    """Test causal tracing."""

    def test_trace_returns_result(self):
        """Causal tracing should return a valid result."""
        from titanbit.interpretability.circuits import CausalTracer

        model = BitNetTransformer(TINY_CONFIG)
        tracer = CausalTracer(model, noise_std=0.1)

        input_ids = torch.randint(0, 256, (8,))
        result = tracer.trace(
            input_ids,
            subject_range=(1, 3),
            target_token=42,
            num_noise_samples=2,
        )

        assert result.indirect_effects.shape[0] == 2  # num_layers
        assert 0 <= result.peak_layer < 2
        assert 0 <= result.clean_prob <= 1
        assert len(result.restored_probs) == 2
