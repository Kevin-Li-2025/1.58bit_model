"""
titanbit.serving
~~~~~~~~~~~~~~~~
High-performance inference engine for BitNet b1.58 models.

Implements:
    - Autoregressive generation with KV cache
    - Speculative decoding (draft + target verification)
    - Streaming token generation
    - Batch inference with dynamic batching
"""

from titanbit.serving.engine import InferenceEngine, GenerationConfig
from titanbit.serving.speculative import SpeculativeDecoder

__all__ = ["InferenceEngine", "GenerationConfig", "SpeculativeDecoder"]
