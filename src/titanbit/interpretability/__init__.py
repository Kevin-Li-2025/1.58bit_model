"""
titanbit.interpretability
~~~~~~~~~~~~~~~~~~~~~~~~~
Mechanistic interpretability toolkit for BitNet b1.58 models.

Implements:
    - Activation probing (linear probes on hidden states)
    - Causal tracing (activation patching to localise knowledge)
    - Sparse autoencoders for feature decomposition
    - Attention pattern analysis
    - Ternary weight circuit analysis (unique to BitNet)

Why interpretability on ternary models is interesting:
    In standard FP16 models, each weight is a continuous value,
    making circuits hard to analyse.  In a ternary model, each
    "synapse" is either ON (+1), OFF (0), or INVERTED (-1).
    This discrete structure makes circuit analysis more tractable
    and potentially more interpretable.
"""

from titanbit.interpretability.probing import LinearProbe, ProbeTrainer
from titanbit.interpretability.circuits import (
    CausalTracer,
    AttentionAnalyser,
    TernaryCircuitAnalyser,
)

__all__ = [
    "LinearProbe",
    "ProbeTrainer",
    "CausalTracer",
    "AttentionAnalyser",
    "TernaryCircuitAnalyser",
]
