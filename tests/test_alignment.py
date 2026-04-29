"""
Tests for TitanAlign (DPO alignment).
"""

import pytest
import torch

from titanbit.model.config import BitNetConfig
from titanbit.model.transformer import BitNetTransformer
from titanbit.alignment.dpo import DPOTrainer, DPOConfig
from titanbit.alignment.data import PreferenceExample, PreferenceDataset


TINY_CONFIG = BitNetConfig(
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    vocab_size=256,
    max_seq_length=128,
)


class TestDPOLoss:
    """Test DPO loss computation."""

    def test_sigmoid_loss(self):
        """Standard DPO sigmoid loss should be positive."""
        model = BitNetTransformer(TINY_CONFIG)
        config = DPOConfig(beta=0.1, reference_free=True)
        trainer = DPOTrainer(model, config, tokenizer=None)

        chosen_logps = torch.tensor([[-10.0, -8.0]])
        rejected_logps = torch.tensor([[-15.0, -12.0]])

        loss, metrics = trainer.dpo_loss_reference_free(
            chosen_logps.squeeze(), rejected_logps.squeeze()
        )
        assert loss.item() > 0
        assert metrics["accuracy"] > 0

    def test_chosen_preferred(self):
        """When chosen has higher log-prob, accuracy should be 1."""
        model = BitNetTransformer(TINY_CONFIG)
        config = DPOConfig(beta=0.1, reference_free=True)
        trainer = DPOTrainer(model, config, tokenizer=None)

        chosen = torch.tensor([-5.0])
        rejected = torch.tensor([-20.0])

        _, metrics = trainer.dpo_loss_reference_free(chosen, rejected)
        assert metrics["accuracy"] == 1.0

    def test_full_dpo_loss(self):
        """Full DPO loss with reference model."""
        model = BitNetTransformer(TINY_CONFIG)
        config = DPOConfig(beta=0.1)
        trainer = DPOTrainer(model, config, tokenizer=None)

        policy_c = torch.tensor([-5.0, -6.0])
        policy_r = torch.tensor([-10.0, -12.0])
        ref_c = torch.tensor([-5.5, -6.5])
        ref_r = torch.tensor([-10.5, -12.5])

        loss, metrics = trainer.dpo_loss(policy_c, policy_r, ref_c, ref_r)
        assert loss.item() > 0
        assert "reward_margin" in metrics

    def test_hinge_loss(self):
        """Hinge DPO variant."""
        model = BitNetTransformer(TINY_CONFIG)
        config = DPOConfig(beta=0.1, loss_type="hinge")
        trainer = DPOTrainer(model, config, tokenizer=None)

        loss, _ = trainer.dpo_loss(
            torch.tensor([-5.0]), torch.tensor([-10.0]),
            torch.tensor([-5.5]), torch.tensor([-10.5]),
        )
        assert loss.item() >= 0

    def test_ipo_loss(self):
        """IPO loss variant."""
        model = BitNetTransformer(TINY_CONFIG)
        config = DPOConfig(beta=0.1, loss_type="ipo")
        trainer = DPOTrainer(model, config, tokenizer=None)

        loss, _ = trainer.dpo_loss(
            torch.tensor([-5.0]), torch.tensor([-10.0]),
            torch.tensor([-5.5]), torch.tensor([-10.5]),
        )
        assert loss.item() >= 0


class TestLogProbs:
    """Test log-probability computation."""

    def test_logprob_shape(self):
        """Log-probs should have shape (B,)."""
        model = BitNetTransformer(TINY_CONFIG)
        config = DPOConfig(beta=0.1, reference_free=True)
        trainer = DPOTrainer(model, config, tokenizer=None)

        B, T = 2, 32
        input_ids = torch.randint(0, 256, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :16] = -100  # mask prompt

        logps = trainer.compute_logprobs(model, input_ids, attention_mask, labels)
        assert logps.shape == (B,)

    def test_logprob_negative(self):
        """Log-probabilities should be negative."""
        model = BitNetTransformer(TINY_CONFIG)
        config = DPOConfig(beta=0.1, reference_free=True)
        trainer = DPOTrainer(model, config, tokenizer=None)

        input_ids = torch.randint(0, 256, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :16] = -100

        logps = trainer.compute_logprobs(model, input_ids, attention_mask, labels)
        assert (logps < 0).all()


class TestPreferenceData:
    """Test preference dataset."""

    def test_example_creation(self):
        ex = PreferenceExample(
            prompt="What is 2+2?",
            chosen="4",
            rejected="5",
        )
        assert ex.prompt == "What is 2+2?"
        assert ex.chosen == "4"
        assert ex.rejected == "5"
