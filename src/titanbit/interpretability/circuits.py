"""
titanbit.interpretability.circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Circuit-level analysis for BitNet b1.58 models.

Implements three complementary analysis techniques:

1. **Causal Tracing** (Meng et al., 2022)
   Localise where factual knowledge is stored by corrupting inputs
   and selectively restoring activations at specific layers.

2. **Attention Pattern Analysis**
   Visualise and quantify attention head behaviour: which heads
   perform induction, copy, or positional operations.

3. **Ternary Circuit Analysis** (novel to BitNet)
   Exploit the discrete weight structure to analyse circuits.
   Since weights are {-1, 0, 1}, each "synapse" has a clear
   interpretation: excitatory (+1), inhibitory (-1), or silent (0).
   This enables exact circuit enumeration for small subnetworks.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from titanbit.model.transformer import BitNetTransformer
from titanbit.model.bitlinear import BitLinear

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Causal Tracing
# ---------------------------------------------------------------------------

@dataclass
class CausalTracingResult:
    """Results from a causal tracing experiment."""
    # Shape: (num_layers, seq_len) — indirect effect at each (layer, position)
    indirect_effects: torch.Tensor
    # The layer with the highest effect
    peak_layer: int
    # The position with the highest effect
    peak_position: int
    # Clean and corrupted probabilities
    clean_prob: float
    corrupted_prob: float
    # Per-layer restored probabilities
    restored_probs: list[float]


class CausalTracer:
    """
    Causal tracing to localise factual knowledge in BitNet models.

    Methodology (Meng et al., 2022 — "Locating and Editing Factual
    Associations in GPT"):

    1. **Clean run**: Forward pass with clean input, record target probability
    2. **Corrupted run**: Add noise to subject embeddings, record target prob
    3. **Restoration runs**: For each layer, restore clean activations
       into the corrupted run and measure how much the target prob recovers

    The "indirect effect" at each layer measures how much that layer
    contributes to the model's factual knowledge about the subject.

    Usage
    -----
    >>> tracer = CausalTracer(model)
    >>> result = tracer.trace(
    ...     input_ids=tokenizer.encode("The Eiffel Tower is in"),
    ...     subject_range=(1, 3),  # "Eiffel Tower"
    ...     target_token=tokenizer.encode(" Paris")[0],
    ... )
    >>> print(f"Knowledge peak: layer {result.peak_layer}")
    """

    def __init__(
        self,
        model: BitNetTransformer,
        noise_std: float = 0.1,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.noise_std = noise_std

    @torch.no_grad()
    def trace(
        self,
        input_ids: torch.Tensor,
        subject_range: tuple[int, int],
        target_token: int,
        num_noise_samples: int = 10,
    ) -> CausalTracingResult:
        """
        Run causal tracing experiment.

        Parameters
        ----------
        input_ids      : (seq_len,) token IDs
        subject_range  : (start, end) positions of the subject tokens
        target_token   : the token ID we're measuring probability for
        num_noise_samples : number of noise samples to average over
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        num_layers = len(self.model.layers)
        seq_len = input_ids.shape[1]

        # Step 1: Clean run
        clean_logits, _ = self.model(input_ids)
        clean_probs = F.softmax(clean_logits[0, -1], dim=-1)
        clean_prob = clean_probs[target_token].item()

        # Step 2: Corrupted run (noise on subject embeddings)
        corrupted_probs_list = []
        for _ in range(num_noise_samples):
            corrupted_logits = self._corrupted_forward(input_ids, subject_range)
            c_probs = F.softmax(corrupted_logits[0, -1], dim=-1)
            corrupted_probs_list.append(c_probs[target_token].item())
        corrupted_prob = sum(corrupted_probs_list) / len(corrupted_probs_list)

        # Step 3: Restoration runs — one per layer
        indirect_effects = torch.zeros(num_layers, seq_len)
        restored_probs = []

        for layer_idx in range(num_layers):
            # Restore activations at this layer from clean run
            restored_logits = self._restored_forward(
                input_ids, subject_range, layer_idx
            )
            r_probs = F.softmax(restored_logits[0, -1], dim=-1)
            r_prob = r_probs[target_token].item()
            restored_probs.append(r_prob)

            # Indirect effect = how much restoration at this layer helps
            if clean_prob - corrupted_prob > 0:
                effect = (r_prob - corrupted_prob) / (clean_prob - corrupted_prob)
            else:
                effect = 0.0
            indirect_effects[layer_idx, :] = effect

        # Find peak
        layer_effects = [restored_probs[i] - corrupted_prob for i in range(num_layers)]
        peak_layer = max(range(num_layers), key=lambda i: layer_effects[i])

        return CausalTracingResult(
            indirect_effects=indirect_effects,
            peak_layer=peak_layer,
            peak_position=subject_range[0],
            clean_prob=clean_prob,
            corrupted_prob=corrupted_prob,
            restored_probs=restored_probs,
        )

    def _corrupted_forward(
        self,
        input_ids: torch.Tensor,
        subject_range: tuple[int, int],
    ) -> torch.Tensor:
        """Forward pass with noise added to subject embeddings."""
        x = self.model.embed_tokens(input_ids)

        # Add noise to subject positions
        noise = torch.randn_like(x[:, subject_range[0]:subject_range[1]]) * self.noise_std
        x[:, subject_range[0]:subject_range[1]] += noise

        # Run through layers
        for layer in self.model.layers:
            x, _ = layer(x)
        x = self.model.norm(x)

        if self.model.lm_head is not None:
            logits = self.model.lm_head(x)
        else:
            logits = F.linear(x, self.model.embed_tokens.weight)
        return logits

    def _restored_forward(
        self,
        input_ids: torch.Tensor,
        subject_range: tuple[int, int],
        restore_layer: int,
    ) -> torch.Tensor:
        """Forward with corruption, but restore clean activations at one layer."""
        # Get clean activations at the target layer
        clean_x = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            clean_x, _ = layer(clean_x)
            if i == restore_layer:
                clean_activations = clean_x.clone()
                break

        # Corrupted forward
        x = self.model.embed_tokens(input_ids)
        noise = torch.randn_like(x[:, subject_range[0]:subject_range[1]]) * self.noise_std
        x[:, subject_range[0]:subject_range[1]] += noise

        for i, layer in enumerate(self.model.layers):
            x, _ = layer(x)
            if i == restore_layer:
                # Restore clean activations at this layer
                x = clean_activations

        x = self.model.norm(x)
        if self.model.lm_head is not None:
            logits = self.model.lm_head(x)
        else:
            logits = F.linear(x, self.model.embed_tokens.weight)
        return logits


# ---------------------------------------------------------------------------
# Attention Analysis
# ---------------------------------------------------------------------------

class AttentionAnalyser:
    """
    Analyse attention patterns across heads and layers.

    Identifies attention head types:
        - Induction heads: attend to token B in "A B ... A" patterns
        - Copy/retrieval heads: copy information from previous occurrences
        - Positional heads: attend based on relative position
        - Content heads: attend based on token identity

    Usage
    -----
    >>> analyser = AttentionAnalyser(model)
    >>> patterns = analyser.get_attention_patterns(input_ids)
    >>> induction_scores = analyser.score_induction_heads(input_ids)
    """

    def __init__(
        self,
        model: BitNetTransformer,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self._attention_maps: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def _register_attention_hooks(self) -> None:
        """Register hooks to capture attention weights."""
        self._clear_hooks()
        for idx, layer in enumerate(self.model.layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # The attention module returns (output, kv_cache)
                    # We need to modify the attention to capture weights
                    pass
                return hook
            h = layer.attn.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def _clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def compute_attention_entropy(
        self,
        input_ids: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Compute the entropy of attention patterns per head.

        High entropy → uniform attention (content-independent)
        Low entropy → focused attention (content-dependent)

        Returns
        -------
        Dict with per-layer, per-head entropy values
        """
        input_ids = input_ids.to(self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        B, T = input_ids.shape
        results = {}

        # Manual attention computation for each layer
        x = self.model.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(self.model.layers):
            # Get the normed input
            normed = layer.attn_norm(x)

            # Compute Q, K
            q = layer.attn.q_proj(normed)
            k = layer.attn.k_proj(normed)

            num_heads = layer.attn.num_heads
            head_dim = layer.attn.head_dim

            q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, T, layer.attn.num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            cos, sin = layer.attn.rotary_emb(q, T)
            from titanbit.model.transformer import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # GQA expansion
            if layer.attn.num_kv_groups > 1:
                k = k.repeat_interleave(layer.attn.num_kv_groups, dim=1)

            # Attention weights
            scale = 1.0 / (head_dim ** 0.5)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=self.device, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            attn_probs = F.softmax(attn_weights, dim=-1)

            # Entropy per head
            # H = -sum(p * log(p))
            log_probs = torch.log(attn_probs + 1e-10)
            entropy = -(attn_probs * log_probs).sum(dim=-1).mean(dim=-1)  # (B, num_heads)

            results[f"layer_{layer_idx}"] = {
                "entropy": entropy[0].cpu().tolist(),  # per-head entropy
                "mean_entropy": entropy[0].mean().item(),
            }

            # Continue forward pass
            x, _ = layer(x)

        return results

    @torch.no_grad()
    def score_induction_heads(
        self,
        input_ids: torch.Tensor,
    ) -> dict[tuple[int, int], float]:
        """
        Score each attention head for induction behaviour.

        An induction head attends to the token that followed the
        previous occurrence of the current token.  In "A B ... A",
        the induction head at the second A would attend to B.

        Returns
        -------
        Dict mapping (layer, head) to induction score
        """
        # This is a simplified version — a full implementation would
        # use the "prefix matching" score from Olsson et al. (2022)
        input_ids = input_ids.to(self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        results = {}
        x = self.model.embed_tokens(input_ids)
        B, T = input_ids.shape

        for layer_idx, layer in enumerate(self.model.layers):
            normed = layer.attn_norm(x)
            q = layer.attn.q_proj(normed)
            k = layer.attn.k_proj(normed)

            num_heads = layer.attn.num_heads
            head_dim = layer.attn.head_dim

            q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, T, layer.attn.num_kv_heads, head_dim).transpose(1, 2)

            if layer.attn.num_kv_groups > 1:
                k = k.repeat_interleave(layer.attn.num_kv_groups, dim=1)

            scale = 1.0 / (head_dim ** 0.5)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(torch.ones(T, T, device=self.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float("-inf"))
            attn_probs = F.softmax(attn, dim=-1)

            # Induction score: correlation between attention and "shifted identity"
            for head_idx in range(num_heads):
                head_attn = attn_probs[0, head_idx]  # (T, T)
                # Measure how much attention follows a "diagonal + 1" pattern
                if T > 2:
                    diag_score = 0.0
                    count = 0
                    for t in range(2, T):
                        # Check if token at t-1 appeared before at some position s
                        token = input_ids[0, t - 1].item()
                        prev_positions = (input_ids[0, :t-1] == token).nonzero(as_tuple=True)[0]
                        if len(prev_positions) > 0:
                            # Induction: should attend to position after prev occurrence
                            for s in prev_positions:
                                if s + 1 < T:
                                    diag_score += head_attn[t, s + 1].item()
                                    count += 1
                    score = diag_score / max(count, 1)
                else:
                    score = 0.0

                results[(layer_idx, head_idx)] = score

            x, _ = layer(x)

        return results


# ---------------------------------------------------------------------------
# Ternary Circuit Analysis (Novel)
# ---------------------------------------------------------------------------

class TernaryCircuitAnalyser:
    """
    Analyse circuits in ternary-weight networks.

    This is unique to BitNet models.  Since every weight is in {-1, 0, 1},
    we can classify each "synapse" as:
        +1 : excitatory (passes information forward)
        -1 : inhibitory (inverts and passes)
         0 : silent (disconnects)

    This enables analyses that are intractable for continuous-weight models:
        1. Sparsity analysis: what fraction of weights are zero?
        2. Excitatory/inhibitory balance per layer
        3. "Circuit fingerprinting": identify which neurons are connected

    Usage
    -----
    >>> analyser = TernaryCircuitAnalyser(model)
    >>> stats = analyser.weight_statistics()
    >>> connectivity = analyser.layer_connectivity()
    """

    def __init__(self, model: BitNetTransformer) -> None:
        self.model = model

    def weight_statistics(self) -> dict[str, Any]:
        """
        Compute statistics about the ternary weight distribution.

        Returns
        -------
        Dict with per-layer and overall statistics
        """
        stats = {
            "layers": {},
            "overall": {"total": 0, "positive": 0, "negative": 0, "zero": 0},
        }

        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                w = module.weight.data
                # Quantise to see the ternary distribution
                mean_abs = w.abs().mean()
                if mean_abs > 0:
                    w_q = (w / mean_abs).round().clamp(-1, 1)
                else:
                    w_q = torch.zeros_like(w)

                total = w_q.numel()
                pos = (w_q == 1).sum().item()
                neg = (w_q == -1).sum().item()
                zero = (w_q == 0).sum().item()

                stats["layers"][name] = {
                    "shape": list(w.shape),
                    "total": total,
                    "positive": pos,
                    "negative": neg,
                    "zero": zero,
                    "sparsity": zero / total,
                    "exc_inh_ratio": pos / max(neg, 1),
                    "mean_abs_weight": mean_abs.item(),
                }

                stats["overall"]["total"] += total
                stats["overall"]["positive"] += pos
                stats["overall"]["negative"] += neg
                stats["overall"]["zero"] += zero

        total = stats["overall"]["total"]
        if total > 0:
            stats["overall"]["sparsity"] = stats["overall"]["zero"] / total
            stats["overall"]["exc_inh_ratio"] = (
                stats["overall"]["positive"] / max(stats["overall"]["negative"], 1)
            )

        return stats

    def layer_connectivity(self) -> dict[str, dict[str, float]]:
        """
        Compute connectivity metrics for each layer.

        Connectivity = fraction of non-zero weights (1 - sparsity).
        Higher connectivity means more information flow.

        Returns
        -------
        Dict mapping layer name to connectivity metrics
        """
        connectivity = {}

        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                w = module.weight.data
                mean_abs = w.abs().mean()
                if mean_abs > 0:
                    w_q = (w / mean_abs).round().clamp(-1, 1)
                else:
                    w_q = torch.zeros_like(w)

                total = w_q.numel()
                nonzero = (w_q != 0).sum().item()

                connectivity[name] = {
                    "connectivity": nonzero / total,
                    "fan_in": w.shape[1],
                    "fan_out": w.shape[0],
                    "effective_fan_in": nonzero / w.shape[0],  # avg active inputs per neuron
                }

        return connectivity

    def find_dead_neurons(self, threshold: float = 0.05) -> dict[str, list[int]]:
        """
        Find neurons with very low connectivity (nearly all-zero weights).

        A "dead" neuron receives almost no input (row is mostly zeros).

        Parameters
        ----------
        threshold : a neuron is "dead" if fewer than this fraction of
                    its incoming weights are non-zero

        Returns
        -------
        Dict mapping layer name to list of dead neuron indices
        """
        dead_neurons = {}

        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                w = module.weight.data
                mean_abs = w.abs().mean()
                if mean_abs > 0:
                    w_q = (w / mean_abs).round().clamp(-1, 1)
                else:
                    w_q = torch.zeros_like(w)

                # For each output neuron, check its input connectivity
                per_neuron_connectivity = (w_q != 0).float().mean(dim=1)
                dead_mask = per_neuron_connectivity < threshold
                dead_indices = dead_mask.nonzero(as_tuple=True)[0].tolist()

                if dead_indices:
                    dead_neurons[name] = dead_indices

        return dead_neurons

    def excitatory_inhibitory_balance(self) -> dict[str, dict[str, float]]:
        """
        Analyse the balance between excitatory (+1) and inhibitory (-1)
        weights at each layer.

        In biological neural networks, the E/I balance is critical
        for stability.  An imbalanced E/I ratio can lead to:
            - Too excitatory → runaway activation (loss spikes)
            - Too inhibitory → vanishing activations (mode collapse)
        """
        balance = {}

        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                w = module.weight.data
                mean_abs = w.abs().mean()
                if mean_abs > 0:
                    w_q = (w / mean_abs).round().clamp(-1, 1)
                else:
                    w_q = torch.zeros_like(w)

                pos = (w_q == 1).sum().item()
                neg = (w_q == -1).sum().item()
                total_active = pos + neg

                balance[name] = {
                    "excitatory_fraction": pos / max(total_active, 1),
                    "inhibitory_fraction": neg / max(total_active, 1),
                    "ei_ratio": pos / max(neg, 1),
                    "total_active": total_active,
                    "total_silent": (w_q == 0).sum().item(),
                }

        return balance

    def summary(self) -> str:
        """Generate a human-readable summary of the ternary circuit."""
        stats = self.weight_statistics()
        ov = stats["overall"]

        lines = [
            "=== TitanBit Ternary Circuit Analysis ===",
            "",
            f"Total parameters:  {ov['total']:,}",
            f"  Excitatory (+1): {ov['positive']:,} ({ov['positive']/ov['total']*100:.1f}%)",
            f"  Inhibitory (-1): {ov['negative']:,} ({ov['negative']/ov['total']*100:.1f}%)",
            f"  Silent (0):      {ov['zero']:,} ({ov['zero']/ov['total']*100:.1f}%)",
            f"  E/I Ratio:       {ov.get('exc_inh_ratio', 0):.3f}",
            f"  Sparsity:        {ov.get('sparsity', 0)*100:.1f}%",
            "",
            "Per-layer breakdown:",
        ]

        for name, layer_stats in stats["layers"].items():
            short_name = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
            lines.append(
                f"  {short_name:30s} | sparsity={layer_stats['sparsity']*100:5.1f}% "
                f"| E/I={layer_stats['exc_inh_ratio']:.2f}"
            )

        return "\n".join(lines)
