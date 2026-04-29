"""
titanbit.cli
~~~~~~~~~~~~~
Command-line interface for TitanBit.

Commands:
    titanbit train    — Run pretraining
    titanbit eval     — Evaluate a checkpoint
    titanbit bench    — Benchmark Triton kernels
    titanbit tokenize — Tokenise a corpus to binary format
    titanbit info     — Show model architecture info
"""

from __future__ import annotations

import logging
import sys

import click
import yaml

logger = logging.getLogger(__name__)


def safe_echo(msg: str) -> None:
    """Echo that gracefully handles Windows encoding issues with emoji."""
    try:
        click.echo(msg)
    except UnicodeEncodeError:
        click.echo(msg.encode("ascii", errors="replace").decode("ascii"))


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def main(log_level: str) -> None:
    """TitanBit — 1.58-bit LLM pretraining engine."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--dataset", default="HuggingFaceFW/fineweb", help="HF dataset name")
@click.option("--subset", default="sample-10BT", help="Dataset subset")
@click.option("--output", required=True, help="Path to save the .bin file")
@click.option("--max-tokens", default=1_000_000_000, type=int, help="Max tokens to download")
def download(dataset, subset, output, max_tokens):
    """Download and pre-tokenise a dataset for offline training."""
    from titanbit.training.data import DatasetDownloader
    
    downloader = DatasetDownloader(dataset_name=dataset, subset=subset)
    downloader.download_and_tokenize(output, max_tokens=max_tokens)
    safe_echo(f"\nDone! You can now train using your config with train_data_path: {output}")


@main.command()
@click.option("--config", "-c", required=True, help="Path to YAML config file")
@click.option("--resume", default="", help="Path to checkpoint to resume from")
def train(config: str, resume: str) -> None:
    """Run pretraining."""
    from titanbit.model.config import BitNetConfig
    from titanbit.training.trainer import Trainer, TrainerConfig

    with open(config) as f:
        raw = yaml.safe_load(f)

    model_cfg = BitNetConfig.from_dict(raw.get("model", {}))
    train_cfg = TrainerConfig.from_dict(raw.get("training", {}))

    if resume:
        train_cfg.resume_from = resume

    safe_echo(f"TitanBit Training -- {model_cfg.num_params_str} model")
    safe_echo(f"   Config: {config}")

    trainer = Trainer(model_cfg, train_cfg)
    stats = trainer.train()

    safe_echo(f"\nTraining complete! {stats['total_steps']} steps, "
              f"{stats['tokens_processed']:,} tokens")


@main.command()
@click.option("--model-size", "-s", default="1.3B",
              type=click.Choice(["125M", "350M", "700M", "1.3B", "3B"]),
              help="Pre-defined model size")
def info(model_size: str) -> None:
    """Show model architecture details."""
    from titanbit.model.config import MODEL_REGISTRY

    cfg = MODEL_REGISTRY[model_size]

    safe_echo(f"\nBitNet b1.58 -- {model_size}")
    safe_echo(f"   Parameters:      {cfg.num_params:,} ({cfg.num_params_str})")
    safe_echo(f"   Hidden size:     {cfg.hidden_size}")
    safe_echo(f"   Layers:          {cfg.num_layers}")
    safe_echo(f"   Heads:           {cfg.num_heads} (KV: {cfg.num_kv_heads})")
    safe_echo(f"   Head dim:        {cfg.head_dim}")
    safe_echo(f"   Intermediate:    {cfg.intermediate_size}")
    safe_echo(f"   Vocab size:      {cfg.vocab_size}")
    safe_echo(f"   Max seq length:  {cfg.max_seq_length}")
    safe_echo(f"   MLP type:        {cfg.mlp_type}")
    safe_echo(f"   Weight bits:     {cfg.weight_bits}")
    safe_echo(f"   Activation bits: {cfg.activation_bits}")

    # Memory estimate
    param_bytes_bf16 = cfg.num_params * 2
    optimizer_bytes = cfg.num_params * 8  # AdamW: 2 states × 4 bytes
    grad_bytes = cfg.num_params * 2
    total_static = param_bytes_bf16 + optimizer_bytes + grad_bytes
    safe_echo(f"\nMemory estimates (BF16 training):")
    safe_echo(f"   Weights:    {param_bytes_bf16 / 1e9:.2f} GB")
    safe_echo(f"   Optimizer:  {optimizer_bytes / 1e9:.2f} GB")
    safe_echo(f"   Gradients:  {grad_bytes / 1e9:.2f} GB")
    safe_echo(f"   Static:     {total_static / 1e9:.2f} GB")
    safe_echo(f"   ~Activations: 5-15 GB (seq_len dependent)")
    safe_echo(f"   ~Total:     {total_static / 1e9 + 10:.0f} GB")


@main.command()
@click.option("--m", default=2048, help="M dimension")
@click.option("--k", default=2048, help="K dimension")
@click.option("--n", default=2048, help="N dimension")
def bench(m: int, k: int, n: int) -> None:
    """Benchmark Triton ternary matmul kernel."""
    from titanbit.model.kernels import benchmark_ternary_matmul

    safe_echo(f"Benchmarking ternary matmul: [{m} x {k}] @ [{k} x {n}]")
    results = benchmark_ternary_matmul(M=m, K=k, N=n)

    if results:
        safe_echo(f"\n   cuBLAS:  {results.get('cublas_ms', 0):.3f} ms "
                  f"({results.get('cublas_tflops', 0):.2f} TFLOPS)")
        safe_echo(f"   Triton:  {results.get('triton_ms', 0):.3f} ms "
                  f"({results.get('triton_tflops', 0):.2f} TFLOPS)")
        safe_echo(f"   Speedup: {results.get('speedup', 0):.2f}x")
    else:
        safe_echo("   CUDA not available")


@main.command()
@click.option("--input", "-i", required=True, help="Input text file or directory")
@click.option("--output", "-o", required=True, help="Output .bin file path")
@click.option("--tokenizer", default="tiktoken:gpt2", help="Tokenizer to use")
@click.option("--val-split", default=0.01, help="Fraction for validation set")
def tokenize(input: str, output: str, tokenizer: str, val_split: float) -> None:
    """Tokenise a text corpus into binary format for training."""
    import os
    from titanbit.training.data import tokenize_and_save

    safe_echo(f"Tokenising {input} -> {output}")

    # Read input
    if os.path.isdir(input):
        texts = []
        for f in sorted(os.listdir(input)):
            fp = os.path.join(input, f)
            if os.path.isfile(fp):
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    texts.append(fh.read())
    else:
        with open(input, "r", encoding="utf-8", errors="ignore") as f:
            texts = [f.read()]

    # Split train/val
    if val_split > 0:
        split_idx = max(1, int(len(texts) * (1 - val_split)))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
    else:
        train_texts = texts
        val_texts = []

    # Tokenise
    train_meta = tokenize_and_save(train_texts, output, tokenizer)
    safe_echo(f"   Train: {train_meta['num_tokens']:,} tokens")

    if val_texts:
        val_output = output.replace(".bin", "_val.bin")
        val_meta = tokenize_and_save(val_texts, val_output, tokenizer)
        safe_echo(f"   Val:   {val_meta['num_tokens']:,} tokens")


if __name__ == "__main__":
    main()
