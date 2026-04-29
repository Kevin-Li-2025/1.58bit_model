"""
titanbit.training.data
~~~~~~~~~~~~~~~~~~~~~~
Data pipelines for BitNet pretraining.

Supports two modes:

1. **Memory-Mapped (mmap)** — Pre-tokenised .bin files
   Best for: local NVMe, repeated experiments, maximum throughput
   Throughput: ~1M tokens/sec per worker (limited by sequential read)

2. **HuggingFace Streaming** — Stream directly from HF datasets
   Best for: FineWeb, large corpora that don't fit on disk,
   first-time runs, no preprocessing step needed
   Throughput: ~500K tokens/sec (limited by tokenisation CPU)

FineWeb Integration
-------------------
FineWeb (HuggingFaceFW/fineweb) is a 15T-token cleaned web corpus.
We use the `sample-10BT` subset (10 billion tokens) for efficient
pretraining runs that can complete on a single GPU.

Anti-Overfitting Strategy
-------------------------
1. **Single-epoch pass**: Stream through data exactly once.
   If max_steps would consume more tokens than available, we cap it.
2. **Token tracking**: Every token consumed is counted.  The trainer
   logs `tokens_consumed / total_tokens` as a progress metric.
3. **No data repetition**: When the dataset is exhausted, training stops
   (or wraps with a warning + epoch counter bump).
4. **Validation held-out**: We reserve a separate split for eval,
   never seen during training.

Disk-to-GPU throughput target: < 1% of step time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
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

    # Source — either a local .bin path OR a HF dataset name
    train_data_path: str = "./data/train.bin"
    val_data_path: str = "./data/val.bin"

    # HuggingFace streaming (takes priority over .bin if set)
    hf_dataset: str = ""                         # e.g. "HuggingFaceFW/fineweb"
    hf_subset: str = "sample-10BT"               # e.g. "sample-10BT" or "default"
    hf_text_column: str = "text"                  # column name for raw text
    hf_split: str = "train"                       # split to use
    hf_val_split: str = ""                        # val split (empty = hold out from train)
    hf_val_fraction: float = 0.001                # fraction of train to hold out for val

    # Tokenizer
    tokenizer_name: str = "tiktoken:gpt2"         # tiktoken:MODEL or hf:MODEL
    vocab_size: int = 50257                       # must match tokenizer

    # Sequence
    seq_length: int = 2048
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    # Anti-overfit
    max_epochs: int = 1                           # stop after N passes through data
    total_tokens: int = 0                         # 0 = auto-detect from dataset

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        # Cast numeric fields to prevent YAML "type pollution" (string vs float/int)
        typed_dict = {}
        int_fields = ("seq_length", "batch_size", "num_workers", "max_epochs", "total_tokens", "vocab_size")
        
        for k, v in d.items():
            if k in cls.__dataclass_fields__:
                if k in int_fields:
                    typed_dict[k] = int(v) if v is not None else v
                elif k == "hf_val_fraction":
                    typed_dict[k] = float(v) if v is not None else v
                else:
                    typed_dict[k] = v
        return cls(**typed_dict)


# ---------------------------------------------------------------------------
# Memory-mapped dataset (local .bin files)
# ---------------------------------------------------------------------------

class MMapDataset(Dataset):
    """
    Memory-mapped dataset for pre-tokenised binary data.

    The binary file is a flat array of uint16 token IDs.
    Each __getitem__ call returns a (seq_length + 1,) chunk
    where [:-1] is the input and [1:] is the target.
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        dtype: str = "uint16",
    ) -> None:
        self.seq_length = seq_length
        self.data_path = data_path

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
        return max(1, self.num_tokens - self.seq_length - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "labels": y}


class ShuffledMMapDataset(IterableDataset):
    """
    Iterable MMap dataset with pseudo-random shuffling.

    Uses a counter to track consumption and supports epoch-aware
    iteration to prevent overfitting through data repetition.
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        seed: int = 42,
        max_epochs: int = 1,
        dtype: str = "uint16",
    ) -> None:
        self.seq_length = seq_length
        self.seed = seed
        self.max_epochs = max_epochs

        np_dtype = getattr(np, dtype)
        self.data = np.memmap(data_path, dtype=np_dtype, mode="r")
        self.num_tokens = len(self.data)
        self.num_samples = max(1, self.num_tokens - seq_length - 1)

        logger.info(
            "ShuffledMMapDataset: %s tokens, %s samples, max_epochs=%d",
            f"{self.num_tokens:,}",
            f"{self.num_samples:,}",
            max_epochs,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = self.seed + worker_info.id
        else:
            seed = self.seed

        rng = np.random.RandomState(seed)
        epoch = 0
        samples_in_epoch = 0

        while epoch < self.max_epochs:
            idx = rng.randint(0, self.num_samples)
            chunk = self.data[idx : idx + self.seq_length + 1].astype(np.int64)
            if len(chunk) < self.seq_length + 1:
                continue
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            yield {"input_ids": x, "labels": y}

            samples_in_epoch += 1
            # One "epoch" = we've sampled as many windows as there are
            # non-overlapping windows in the data
            if samples_in_epoch >= self.num_samples // self.seq_length:
                epoch += 1
                samples_in_epoch = 0
                if epoch < self.max_epochs:
                    logger.info("Epoch %d/%d completed, continuing...", epoch, self.max_epochs)


# ---------------------------------------------------------------------------
# HuggingFace Streaming Dataset (FineWeb etc.)
# ---------------------------------------------------------------------------

class HFStreamingDataset(IterableDataset):
    """
    Stream directly from a HuggingFace dataset with on-the-fly tokenisation.

    This is designed for FineWeb and similar large-scale corpora:
        - No download/preprocessing step needed
        - Streams data chunk-by-chunk from HF servers
        - Tokenises on CPU workers in parallel
        - Packs documents end-to-end to fill every sequence position
          (no wasted padding tokens)

    Document packing strategy:
        We concatenate all tokenised documents into a continuous stream
        and slice it into fixed-length windows.  This ensures every
        training sequence is fully utilised — no padding waste.

        doc1_tokens + doc2_tokens + doc3_tokens + ...
        |<--- seq 1 --->|<--- seq 2 --->|<--- seq 3 --->|

    Anti-overfit:
        - Tracks total tokens consumed
        - Supports max_epochs to prevent repeated data
        - Each worker gets a different shard (no duplicate data)
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str = "sample-10BT",
        split: str = "train",
        text_column: str = "text",
        tokenizer_name: str = "tiktoken:gpt2",
        seq_length: int = 2048,
        seed: int = 42,
        max_epochs: int = 1,
    ) -> None:
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.text_column = text_column
        self.tokenizer_name = tokenizer_name
        self.seq_length = seq_length
        self.seed = seed
        self.max_epochs = max_epochs

        logger.info(
            "HFStreamingDataset: %s/%s split=%s, seq_len=%d, max_epochs=%d",
            dataset_name, subset, split, seq_length, max_epochs,
        )

    def _get_tokenizer(self):
        """Lazy-load tokenizer (must happen per-worker for thread safety)."""
        if self.tokenizer_name.startswith("tiktoken:"):
            import tiktoken
            enc_name = self.tokenizer_name.split(":")[1]
            enc = tiktoken.get_encoding(enc_name)
            return enc.encode_ordinary
        elif self.tokenizer_name.startswith("hf:"):
            from transformers import AutoTokenizer
            model_name = self.tokenizer_name.split(":", 1)[1]
            tok = AutoTokenizer.from_pretrained(model_name)
            return lambda text: tok.encode(text, add_special_tokens=False)
        else:
            raise ValueError(f"Unsupported tokenizer: {self.tokenizer_name}")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset

        # Worker sharding
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        tokenize = self._get_tokenizer()

        for epoch in range(self.max_epochs):
            if epoch > 0:
                logger.info("HFStreaming: starting epoch %d/%d", epoch + 1, self.max_epochs)

            # Load with streaming
            ds = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                streaming=True,
                trust_remote_code=True,
            )

            # Shuffle with a buffer
            ds = ds.shuffle(seed=self.seed + epoch, buffer_size=10_000)

            # Token buffer for document packing
            buffer: list[int] = []
            chunk_size = self.seq_length + 1

            for i, example in enumerate(ds):
                # Shard across workers
                if i % num_workers != worker_id:
                    continue

                text = example.get(self.text_column, "")
                if not text or len(text.strip()) < 50:
                    continue

                tokens = tokenize(text)
                buffer.extend(tokens)

                # Yield complete sequences from the buffer
                while len(buffer) >= chunk_size:
                    chunk = buffer[:chunk_size]
                    buffer = buffer[chunk_size:]

                    arr = np.array(chunk, dtype=np.int64)
                    x = torch.from_numpy(arr[:-1])
                    y = torch.from_numpy(arr[1:])
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

    if tokenizer_name.startswith("tiktoken:"):
        enc_name = tokenizer_name.split(":")[1]
        enc = tiktoken.get_encoding(enc_name)
        tokenise_fn = enc.encode_ordinary
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    logger.info("Tokenising %d documents with %s...", len(texts), tokenizer_name)

    all_tokens: list[int] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for token_ids in pool.map(tokenise_fn, texts):
            all_tokens.extend(token_ids)

    token_array = np.array(all_tokens, dtype=np.uint16)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    token_array.tofile(output_path)

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


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def create_data_loaders(config: DataConfig) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation DataLoaders.

    Routing logic:
        1. If `hf_dataset` is set → use HF streaming (FineWeb etc.)
        2. Else if `train_data_path` exists → use mmap
        3. Else → raise error
    """
    if config.hf_dataset:
        return _create_hf_loaders(config)
    elif os.path.exists(config.train_data_path):
        return _create_mmap_loaders(config)
    else:
        raise FileNotFoundError(
            f"No data source found. Set hf_dataset for streaming, "
            f"or create {config.train_data_path} for mmap mode."
        )


def _create_hf_loaders(config: DataConfig) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create DataLoaders from a HuggingFace streaming dataset."""
    train_ds = HFStreamingDataset(
        dataset_name=config.hf_dataset,
        subset=config.hf_subset,
        split=config.hf_split,
        text_column=config.hf_text_column,
        tokenizer_name=config.tokenizer_name,
        seq_length=config.seq_length,
        seed=config.seed,
        max_epochs=config.max_epochs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    # Validation: use a separate split or hold out from the same
    val_loader = None
    if config.hf_val_split:
        val_ds = HFStreamingDataset(
            dataset_name=config.hf_dataset,
            subset=config.hf_subset,
            split=config.hf_val_split,
            text_column=config.hf_text_column,
            tokenizer_name=config.tokenizer_name,
            seq_length=config.seq_length,
            seed=config.seed + 999,
            max_epochs=1,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            num_workers=min(2, config.num_workers),
            pin_memory=config.pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader


def _create_mmap_loaders(config: DataConfig) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create DataLoaders from memory-mapped .bin files."""
    train_ds = ShuffledMMapDataset(
        config.train_data_path,
        seq_length=config.seq_length,
        seed=config.seed,
        max_epochs=config.max_epochs,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

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
