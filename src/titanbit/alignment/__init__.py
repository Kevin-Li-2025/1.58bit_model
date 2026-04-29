"""
titanbit.alignment
~~~~~~~~~~~~~~~~~~
Post-training alignment for BitNet b1.58 models.

Implements DPO (Direct Preference Optimization) and KTO (Kahneman-Tversky
Optimization) from scratch — no dependency on trl or other alignment
libraries.

This is a research contribution: investigating whether ternary-weight
models can be effectively aligned via preference learning, and what
unique challenges arise from the quantised weight manifold.
"""

from titanbit.alignment.dpo import DPOTrainer, DPOConfig
from titanbit.alignment.data import PreferenceDataset, load_preference_data

__all__ = ["DPOTrainer", "DPOConfig", "PreferenceDataset", "load_preference_data"]
