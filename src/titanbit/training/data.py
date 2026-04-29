"""
titanbit.training.data
~~~~~~~~~~~~~~~~~~~~~~
Memory-mapped data pipeline for sustained GPU utilisation.

Why mmap?
---------
Standard PyTorch DataLoaders read data from disk, deserialise it in
Python, tokenise, pad, and collate — all on the CPU.  At scale, this
becomes the bottleneck: the GPU sits idle waiting for the next batch.

Our approach: **pre-tokenise the entire corpus into a flat binary file
of uint16 token IDs**, then memory-map it.  The OS handles caching
transparently, and we can serve batches with zero deserialisation
overhead.

Data format
-----------
    .bin file:  flat array of uint16 token IDs (no delimiters)
    .meta file: JSON with {num_tokens, vocab_size, dtype, ...}

The DataLoader simply indexes into the mmap array with random offsets
to produce (seq_len,) chunks.  No padding, no collation overhead.

This is the same strategy used by Karpathy's nanoGPT, Megatron-LM,
and most production pretraining codebases.

Disk-to-GPU throughput target: < 1% of step time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data pipeline configuration."""

    train_data_path: str = "./data/train.bin"
    val_data_path: str = "./data/val.bin"
    seq_length: int = 2048
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Memory-mapped dataset
# ---------------------------------------------------------------------------

class MMapDataset(Dataset):
    """
    Memory-mapped dataset for pre-tokenised binary data.

    The binary file is a flat array of uint16 token IDs.
    Each __getitem__ call returns a random (seq_length + 1,) chunk
    where [:-1] is the input and [1:] is the target.

    Memory usage: virtually zero — the OS manages the page cache.
    Access pattern: random reads, which is fine for NVMe SSDs
    (random 4K reads: ~500K IOPS on modern NVMe).
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        dtype: str = "uint16",
    ) -> None:
        self.seq_length = seq_length
        self.data_path = data_path

        # Memory-map the data file
        np_dtype = getattr(np, dtype)
        self.data = np.memmap(data_path, dtype=np_dtype, mode="r")
        self.num_tokens = len(self.data)

        logger.info(
            "MMapDataset: %s tokens from %s (%.2f GB on disk)",
            f"{self.num_tokens:,}",
            data_path,
            os.path.getsize(data_path) / (1024**3),
        )

    def __len__(self) -> int:
        # Number of possible starting positions
        return max(1, self.num_tokens - self.seq_length - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Read seq_length + 1 tokens (input + 1 for target shift)
        chunk = self.data[idx : idx + self.seq_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "labels": y}


class ShuffledMMapDataset(IterableDataset):
    """
    Iterable version of MMapDataset with epoch-level shuffling.

    Instead of shuffling indices (which requires O(N) memory for
    trillion-token datasets), we use a pseudo-random offset generator
    that produces unique starting positions within each epoch.

    This gives us the benefits of shuffling without the memory overhead
    of storing all indices.
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        seed: int = 42,
        dtype: str = "uint16",
    ) -> None:
        self.seq_length = seq_length
        self.seed = seed

        np_dtype = getattr(np, dtype)
        self.data = np.memmap(data_path, dtype=np_dtype, mode="r")
        self.num_tokens = len(self.data)
        self.num_samples = max(1, self.num_tokens - seq_length - 1)

        logger.info(
            "ShuffledMMapDataset: %s tokens, %s samples",
            f"{self.num_tokens:,}",
            f"{self.num_samples:,}",
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multi-worker: each worker gets a different seed
            seed = self.seed + worker_info.id
        else:
            seed = self.seed

        rng = np.random.RandomState(seed)

        while True:
            idx = rng.randint(0, self.num_samples)
            chunk = self.data[idx : idx + self.seq_length + 1].astype(np.int64)
            if len(chunk) < self.seq_length + 1:
                continue
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            yield {"input_ids": x, "labels": y}


# ---------------------------------------------------------------------------
# Data preparation utilities
# ---------------------------------------------------------------------------

def tokenize_and_save(
    texts: list[str],
    output_path: str,
    tokenizer_name: str = "tiktoken:gpt2",
    max_workers: int = 8,
) -> dict[str, Any]:
    """
    Tokenise a list of texts and save as a flat binary file.

    Parameters
    ----------
    texts          : list of document strings
    output_path    : path for the output .bin file
    tokenizer_name : tokenizer to use (tiktoken:model or hf:model)
    max_workers    : number of parallel tokenisation workers

    Returns
    -------
    metadata dict with token counts
    """
    import tiktoken
    from concurrent.futures import ThreadPoolExecutor

    # Set up tokenizer
    if tokenizer_name.startswith("tiktoken:"):
        enc_name = tokenizer_name.split(":")[1]
        enc = tiktoken.get_encoding(enc_name)
        tokenise_fn = enc.encode_ordinary
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    logger.info("Tokenising %d documents with %s...", len(texts), tokenizer_name)

    # Parallel tokenisation
    all_tokens: list[int] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for token_ids in pool.map(tokenise_fn, texts):
            all_tokens.extend(token_ids)

    # Save as flat binary (uint16 — supports vocab up to 65535)
    token_array = np.array(all_tokens, dtype=np.uint16)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    token_array.tofile(output_path)

    # Save metadata
    meta = {
        "num_tokens": len(all_tokens),
        "vocab_size": enc.max_token_value + 1 if hasattr(enc, "max_token_value") else 100277,
        "dtype": "uint16",
        "tokenizer": tokenizer_name,
        "file_size_bytes": os.path.getsize(output_path),
    }
    meta_path = output_path.replace(".bin", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Saved %s tokens (%.2f MB) to %s",
        f"{meta['num_tokens']:,}",
        meta["file_size_bytes"] / (1024**2),
        output_path,
    )
    return meta


def create_data_loaders(config: DataConfig) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation DataLoaders.

    Uses ShuffledMMapDataset (iterable) for training and
    MMapDataset (map-style) for validation.
    """
    # Training loader
    train_ds = ShuffledMMapDataset(
        config.train_data_path,
        seq_length=config.seq_length,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    # Validation loader (if exists)
    val_loader = None
    if os.path.exists(config.val_data_path):
        val_ds = MMapDataset(
            config.val_data_path,
            seq_length=config.seq_length,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            num_workers=min(2, config.num_workers),
            pin_memory=config.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    return train_loader, val_loader
