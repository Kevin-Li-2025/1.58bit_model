"""
TitanBit — 1.58-bit LLM pretraining engine.

Implements BitNet b1.58 (ternary {-1, 0, 1} weights) with:
  • Custom Triton kernels for ternary matrix multiplication
  • Quantization-Aware Training (QAT) with full-precision shadow weights
  • Memory-mapped data pipeline for sustained GPU utilisation
  • Loss spike detection and automatic recovery
  • MFU (Model Flops Utilisation) tracking
"""

__version__ = "0.1.0"
