<div align="center">

# ⚡ TitanBit

### 1.58-bit LLM Pretraining Engine with Custom Triton Kernels

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org)
[![Triton](https://img.shields.io/badge/Triton-custom%20kernels-purple.svg)](https://triton-lang.org)

*A from-scratch implementation of BitNet b1.58 (ternary {-1, 0, 1} weights) with custom Triton kernels, production-grade training stability, and full observability — designed for single-GPU pretraining on NVIDIA L20 (48GB).*

</div>

---

## 🎯 Why TitanBit?

BitNet b1.58 ([Ma et al., 2024](https://arxiv.org/abs/2402.17764)) represents a paradigm shift: **every weight in the transformer is constrained to {-1, 0, 1}**, encoding each parameter in just log₂(3) ≈ 1.58 bits.  This eliminates floating-point multiplication from the forward pass entirely — the matmul reduces to addition and subtraction.

**TitanBit** is not a wrapper around someone else's library.  It is a **ground-up implementation** demonstrating:

| Component | What it demonstrates |
|-----------|---------------------|
| **BitLinear layer** | Quantisation-Aware Training with STE, AbsMean activation quantisation |
| **Triton kernels** | Custom GPU kernels for ternary matmul (branch-free add/sub) |
| **Weight packing** | 2-bit encoding → 16× memory compression for inference |
| **Full transformer** | RoPE, GQA, SwiGLU, RMSNorm — all with ternary projections |
| **mmap data pipeline** | Zero-copy NVMe reads for sustained GPU utilisation |
| **Stability system** | Loss spike detection, auto-rollback, LR recovery |
| **MFU tracking** | Real-time Model Flops Utilisation against L20 peak (119.5 TFLOPS) |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   BitNet b1.58 Forward Pass                    │
│                                                                │
│  Input x (BF16)                                                │
│      │                                                         │
│      ▼                                                         │
│  ┌────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐   │
│  │RMSNorm │──▶│ AbsMean    │──▶│ Ternary  │──▶│ Rescale  │   │
│  │(SubLN) │   │ Quant (8b) │   │ MatMul   │   │ (β × γ)  │   │
│  └────────┘   └────────────┘   │ {-1,0,1} │   └──────────┘   │
│                                 │ ← STE    │                   │
│                                 └──────────┘                   │
│                                                                │
│  Key: No floating-point multiplication in the matmul.          │
│  The ternary constraint means W×x = ±x or 0.                  │
└────────────────────────────────────────────────────────────────┘
```

### Model Architecture
- **Rotary Position Embeddings (RoPE)** — length generalisation
- **Group Query Attention (GQA)** — reduced KV-cache memory
- **SwiGLU MLP** — ~5% perplexity improvement over GeLU
- **RMSNorm** — pre-norm architecture for stability
- **BitLinear** in all Q/K/V/O and MLP projections

### Pre-defined Sizes

| Size | Hidden | Layers | Heads | Params | L20 VRAM (train) |
|------|--------|--------|-------|--------|------------------|
| 125M | 768 | 12 | 12 | ~125M | ~4 GB |
| 350M | 1024 | 24 | 16 | ~350M | ~8 GB |
| 700M | 1536 | 24 | 24 | ~700M | ~14 GB |
| **1.3B** | **2048** | **24** | **32** | **~1.3B** | **~22 GB** |
| 3B | 3200 | 26 | 32 | ~3B | ~42 GB |

---

## 📦 Installation

```bash
# Core (model + training + Triton kernels)
pip install -e .

# With evaluation benchmarks
pip install -e ".[eval]"

# With FlashAttention
pip install -e ".[flash]"

# Everything
pip install -e ".[all]"
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.2.0 (with CUDA support)
- Triton ≥ 2.2.0
- NVIDIA GPU with compute capability ≥ 7.0 (Ada Lovelace recommended)

---

## 🚀 Quick Start

### 1. Show model info

```bash
titanbit info --model-size 1.3B
```

### 2. Prepare training data

```bash
# Tokenise a text corpus into binary format
titanbit tokenize --input ./data/raw/ --output ./data/train.bin

# Or use TitanWash output directly
titanbit tokenize --input ../TitanWash/data/cleaned/ --output ./data/train.bin
```

### 3. Train

```bash
# Start training with default config (1.3B on L20)
titanbit train --config configs/default.yaml

# Resume from checkpoint
titanbit train --config configs/default.yaml --resume checkpoints/bitnet-1.3B/checkpoint_step_0010000.pt
```

### 4. Benchmark kernels

```bash
# Benchmark Triton ternary matmul vs cuBLAS
titanbit bench --m 2048 --k 2048 --n 2048
```

### Python API

```python
from titanbit.model import BitNetConfig, BitNetTransformer

# Create a 1.3B model
config = BitNetConfig(hidden_size=2048, num_layers=24, num_heads=32)
model = BitNetTransformer(config)

# Forward pass
import torch
ids = torch.randint(0, 32000, (1, 512))
logits, loss = model(ids, labels=ids)
print(f"Loss: {loss.item():.4f}")
```

---

## 🔬 Technical Deep-Dives

### BitLinear: The Core Innovation

Standard `nn.Linear`:  `y = x @ W^T + b`  (FP16 multiply-accumulate)

BitLinear:
```
y = rescale(quant_8bit(RMSNorm(x)) @ quant_ternary(W)^T)
```

1. **RMSNorm (SubLN)** — Normalises input distribution before quantisation
2. **AbsMean Quantisation** — Scales activations to [-128, 127] per-token
3. **Ternary Quantisation** — `W_q = round(W / mean(|W|))` clipped to {-1, 0, 1}
4. **STE** — Gradients bypass the quantisation step during backprop
5. **Rescale** — `out × (β × γ / 127)` restores the magnitude

### Custom Triton Kernel

The ternary matmul kernel eliminates all FMA operations:

```
For each output element y[i,j]:
    y[i,j] = Σ_k  W[j,k] × x[i,k]

    where W[j,k] ∈ {-1, 0, 1}, so:
        if W = +1:  accumulate += x
        if W = -1:  accumulate -= x
        if W =  0:  skip
```

Weight packing: 16 ternary values → 1 int32 (2 bits each)
→ **16× memory compression** for inference weights.

### Stability System

BitNet training is more unstable than standard transformers due to the
non-smooth STE landscape.  TitanBit implements a 4-layer stability system:

```
Layer 1: Gradient clipping (max_norm=1.0)
Layer 2: Loss spike detection (EMA-based, threshold=5×)
Layer 3: Automatic rollback to last stable checkpoint
Layer 4: Learning rate scaling (0.5×) after recovery
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific modules
pytest tests/test_bitlinear.py -v      # Core quantisation
pytest tests/test_model.py -v          # Full transformer

# With coverage
pytest tests/ --cov=titanbit --cov-report=term-missing
```

---

## 📁 Project Structure

```
TitanBit/
├── configs/
│   └── default.yaml              # 1.3B config tuned for L20
├── src/titanbit/
│   ├── model/
│   │   ├── config.py             # Model configs (125M → 3B)
│   │   ├── bitlinear.py          # BitLinear layer + RMSNorm + STE
│   │   ├── transformer.py        # Full transformer (RoPE, GQA, SwiGLU)
│   │   └── kernels.py            # Triton ternary matmul kernel
│   ├── training/
│   │   ├── data.py               # mmap data pipeline
│   │   ├── trainer.py            # Training loop (BF16, MFU, checkpoints)
│   │   └── stability.py          # Loss spike detection & recovery
│   ├── utils/
│   │   └── metrics.py            # GPU metrics, throughput tracking
│   └── cli.py                    # CLI (train, bench, tokenize, info)
├── tests/
│   ├── test_bitlinear.py         # 15+ quantisation tests
│   └── test_model.py             # 10+ transformer tests
├── pyproject.toml                # PEP 621 project config
└── README.md
```

---

## 🔬 References

- **BitNet b1.58** — Ma et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764
- **BitNet** — Wang et al. (2023). *BitNet: Scaling 1-bit Transformers for Large Language Models.* arXiv:2310.11453
- **STE** — Bengio et al. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons.*
- **RoPE** — Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
- **SwiGLU** — Shazeer (2020). *GLU Variants Improve Transformer.*
- **GQA** — Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models.*
- **Chinchilla** — Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.*
- **FlashAttention** — Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention.*

---

## 📝 License

Apache 2.0 — See [LICENSE](LICENSE) for details.
