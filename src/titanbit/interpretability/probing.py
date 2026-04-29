"""
titanbit.interpretability.probing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Linear probing and sparse autoencoders for understanding
what BitNet hidden states encode.

Linear probes test whether a specific concept (e.g., sentiment,
part-of-speech, factual knowledge) is linearly decodable from
the model's internal representations.  If a simple linear classifier
can decode the concept, it suggests the model has learned a
dedicated direction for that feature.

Sparse Autoencoders (SAEs) go further by decomposing the
"superposition" in hidden states — finding the atomic features
that the model uses, even when individual neurons are polysemantic.

These techniques were pioneered by the Anthropic interpretability
team (Cunningham et al., 2023; Bricken et al., 2023).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from titanbit.model.transformer import BitNetTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation Extraction
# ---------------------------------------------------------------------------

class ActivationExtractor:
    """
    Extract activations from specific layers of a BitNet model.

    Uses PyTorch hooks to capture intermediate representations
    without modifying the model code.

    Usage
    -----
    >>> extractor = ActivationExtractor(model, layers=[0, 6, 12, 23])
    >>> output = model(input_ids)
    >>> activations = extractor.get_activations()
    >>> # activations[0] has shape (B, T, hidden_size)
    """

    def __init__(
        self,
        model: BitNetTransformer,
        layers: list[int] | None = None,
        capture_attention: bool = False,
    ) -> None:
        self.model = model
        self.layers = layers or list(range(len(model.layers)))
        self.capture_attention = capture_attention

        self._activations: dict[int, torch.Tensor] = {}
        self._attention_patterns: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        for layer_idx in self.layers:
            if layer_idx >= len(self.model.layers):
                continue

            layer = self.model.layers[layer_idx]

            # Capture output of the full block (after residual)
            def make_hook(idx: int):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self._activations[idx] = output[0].detach()
                    else:
                        self._activations[idx] = output.detach()
                return hook

            h = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def get_activations(self) -> dict[int, torch.Tensor]:
        """Get captured activations (call after forward pass)."""
        return dict(self._activations)

    def clear(self) -> None:
        """Clear captured activations."""
        self._activations.clear()
        self._attention_patterns.clear()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Linear Probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """
    A linear probe for testing what information is encoded in
    hidden states at a specific layer.

    The probe is a simple linear classifier:
        y = softmax(Wx + b)

    If the probe achieves high accuracy, the concept is linearly
    decodable — suggesting the model has learned a dedicated
    representation for it.

    Usage
    -----
    >>> probe = LinearProbe(hidden_size=2048, num_classes=3)  # sentiment
    >>> # Train on (hidden_state, label) pairs
    >>> loss = probe.compute_loss(activations, labels)
    """

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, hidden_size) or (B, T, hidden_size)

        Returns
        -------
        logits : (..., num_classes)
        """
        return self.linear(x)

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for the probe."""
        logits = self.forward(hidden_states)
        if logits.ndim == 3:
            logits = logits.view(-1, self.num_classes)
            labels = labels.view(-1)
        return F.cross_entropy(logits, labels, ignore_index=-100)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        logits = self.forward(hidden_states)
        return logits.argmax(dim=-1)


class ProbeTrainer:
    """
    Train linear probes on model activations.

    Usage
    -----
    >>> trainer = ProbeTrainer(model, probe, layer_idx=12)
    >>> accuracy = trainer.train(train_data, val_data, epochs=10)
    """

    def __init__(
        self,
        model: BitNetTransformer,
        probe: LinearProbe,
        layer_idx: int = -1,
        learning_rate: float = 1e-3,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.probe = probe.to(self.device)
        self.layer_idx = layer_idx
        self.extractor = ActivationExtractor(model, layers=[layer_idx])

        # Freeze the model — only train the probe
        for p in self.model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    @torch.no_grad()
    def extract_features(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run model and extract activations from the target layer."""
        input_ids = input_ids.to(self.device)
        self.extractor.clear()
        _ = self.model(input_ids)
        acts = self.extractor.get_activations()
        return acts.get(self.layer_idx, torch.tensor([]))

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, float]:
        """Single training step. Returns (loss, accuracy)."""
        features = self.extract_features(input_ids)
        if features.numel() == 0:
            return 0.0, 0.0

        labels = labels.to(self.device)
        self.probe.train()

        loss = self.probe.compute_loss(features, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Accuracy
        with torch.no_grad():
            preds = self.probe.predict(features)
            if preds.ndim > 1:
                preds = preds.view(-1)
                flat_labels = labels.view(-1)
            else:
                flat_labels = labels
            mask = flat_labels != -100
            acc = (preds[mask] == flat_labels[mask]).float().mean().item() if mask.any() else 0.0

        return loss.item(), acc


# ---------------------------------------------------------------------------
# Sparse Autoencoder (SAE)
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for decomposing hidden state superposition.

    Architecture:
        h → encoder → ReLU → z (sparse features) → decoder → h_hat

    The sparsity penalty encourages most features to be zero,
    meaning each active feature corresponds to a single interpretable
    concept (monosemantic feature).

    Following Anthropic's approach (Bricken et al., 2023):
        - Expansion factor: 4-16× the hidden dimension
        - L1 sparsity penalty
        - Tied decoder weights (optional)

    Usage
    -----
    >>> sae = SparseAutoencoder(hidden_size=2048, num_features=8192)
    >>> z, h_hat, loss = sae(hidden_states)
    >>> # z contains sparse feature activations
    """

    def __init__(
        self,
        hidden_size: int,
        num_features: int,
        l1_coefficient: float = 1e-3,
        tied_weights: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.l1_coefficient = l1_coefficient
        self.tied_weights = tied_weights

        # Encoder: hidden → features
        self.encoder = nn.Linear(hidden_size, num_features)

        # Decoder: features → hidden (optionally tied)
        if tied_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.decoder = nn.Linear(num_features, hidden_size)

        # Pre-encoder bias (centres the data)
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise with unit-norm decoder columns."""
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to sparse feature activations."""
        x_centred = x - self.pre_bias
        z = F.relu(self.encoder(x_centred))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to hidden space."""
        if self.tied_weights:
            # Use transpose of encoder weights
            h_hat = F.linear(z, self.encoder.weight.t()) + self.decoder_bias
        else:
            h_hat = self.decoder(z)
        return h_hat + self.pre_bias

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        z      : sparse feature activations
        h_hat  : reconstructed hidden states
        loss   : reconstruction loss + L1 sparsity penalty
        """
        z = self.encode(x)
        h_hat = self.decode(z)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(h_hat, x)

        # Sparsity loss (L1 on feature activations)
        sparsity_loss = self.l1_coefficient * z.abs().mean()

        total_loss = recon_loss + sparsity_loss

        return z, h_hat, total_loss

    @property
    def feature_density(self) -> torch.Tensor:
        """Fraction of inputs that activate each feature (after a forward pass)."""
        # This would need to be tracked during training
        return torch.zeros(self.num_features)

    def get_top_features(
        self,
        x: torch.Tensor,
        k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top-k most active features for an input.

        Returns
        -------
        indices : (B, k) indices of top features
        values  : (B, k) activation values
        """
        z = self.encode(x)
        if z.ndim == 3:
            z = z.mean(dim=1)  # average over sequence
        values, indices = torch.topk(z, k, dim=-1)
        return indices, values
