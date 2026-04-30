#!/usr/bin/env python3
"""
download_fineweb_edu.py
========================
Blazing-fast download & pre-tokenisation of the FULL FineWeb-Edu dataset
(1.3 trillion tokens, ~5.8 TB parquet) into memory-mapped .bin shards
ready for TitanBit training.

Speed strategy (fastest → slowest):
────────────────────────────────────
1. `huggingface-cli download` with Xet high-perf backend
   - Multi-threaded Rust-based downloads
   - Saturates any network link (tested 5+ Gbps on datacenter nodes)
   - Downloads raw parquet files to local cache first

2. Parallel tokenisation with multiprocessing + tiktoken
   - N workers tokenise different parquet shards simultaneously
   - Each worker writes to its own .bin shard (no lock contention)
   - Final merge into sequential shards for mmap training

3. Resume support
   - Tracks completed shards in a manifest file
   - Re-running skips already-processed shards

Usage
-----
# Step 1: Download parquet files at max speed (run on server)
python scripts/download_fineweb_edu.py download --output-dir ./data/fineweb-edu-raw

# Step 2: Tokenise into .bin shards
python scripts/download_fineweb_edu.py tokenize \
    --input-dir ./data/fineweb-edu-raw \
    --output-dir ./data \
    --num-workers 16

# Step 3 (optional): Verify
python scripts/download_fineweb_edu.py verify --data-dir ./data

# Or do everything in one shot:
python scripts/download_fineweb_edu.py all --output-dir ./data --num-workers 16

Requirements
------------
pip install huggingface_hub[cli,hf_xet] tiktoken pyarrow tqdm
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fineweb-edu-dl")

# ── Constants ──────────────────────────────────────────────────────────────
DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_TYPE = "dataset"
TOKENIZER_NAME = "gpt2"  # tiktoken encoding
SHARD_MAX_TOKENS = 100_000_000  # 100M tokens per .bin shard (~200 MB)
DTYPE = np.uint16  # 2 bytes per token (vocab < 65k)
VAL_FRACTION = 0.001  # 0.1% held out for validation


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Download raw parquet files at maximum speed
# ═══════════════════════════════════════════════════════════════════════════

def setup_fast_download_env():
    """Configure environment variables for maximum download speed."""
    # Enable Xet high-performance backend (replaces deprecated hf_transfer)
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    # Increase timeout for large files
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1800"
    # Disable symlinks on Windows (if applicable)
    if sys.platform == "win32":
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    logger.info("✅ Fast download environment configured (HF_XET_HIGH_PERFORMANCE=1)")


def download_with_cli(output_dir: str, max_workers: int = 8) -> str:
    """
    Download FineWeb-Edu using huggingface-cli for maximum speed.

    This is the fastest method because:
    - Uses the Rust-based Xet backend for multi-threaded downloads
    - Handles retries and resume automatically
    - Can saturate 10+ Gbps links

    Returns the path to the downloaded dataset directory.
    """
    setup_fast_download_env()
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("📥 DOWNLOADING FineWeb-Edu (FULL — 1.3T tokens, ~5.8 TB)")
    logger.info("   Repo: %s", DATASET_REPO)
    logger.info("   Dest: %s", output_dir)
    logger.info("   Backend: HF Xet High-Performance")
    logger.info("=" * 70)

    # Build the CLI command
    cmd = [
        sys.executable, "-m", "huggingface_hub", "download",
        DATASET_REPO,
        "--repo-type", DATASET_TYPE,
        "--local-dir", output_dir,
        "--include", "data/*/*.parquet",  # Only parquet data files
    ]

    logger.info("Running: %s", " ".join(cmd))
    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output stream to console
        )
    except subprocess.CalledProcessError as e:
        logger.error("❌ Download failed with exit code %d", e.returncode)
        logger.error("   Try running manually: huggingface-cli download %s --repo-type dataset --local-dir %s",
                      DATASET_REPO, output_dir)
        raise

    elapsed = time.time() - start
    logger.info("✅ Download complete in %.1f minutes (%.1f hours)", elapsed / 60, elapsed / 3600)

    return output_dir


def download_with_python_api(output_dir: str) -> str:
    """
    Fallback: Download using the Python API (slower but more portable).
    Uses snapshot_download which also supports resume.
    """
    from huggingface_hub import snapshot_download

    setup_fast_download_env()
    os.makedirs(output_dir, exist_ok=True)

    logger.info("📥 Downloading FineWeb-Edu via Python API...")

    path = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type=DATASET_TYPE,
        local_dir=output_dir,
        allow_patterns="data/*/*.parquet",
        resume_download=True,
    )

    logger.info("✅ Download complete: %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Parallel tokenisation into .bin shards
# ═══════════════════════════════════════════════════════════════════════════

def find_parquet_files(data_dir: str) -> list[str]:
    """Find all parquet files in the downloaded dataset."""
    patterns = [
        os.path.join(data_dir, "data", "*", "*.parquet"),
        os.path.join(data_dir, "**", "*.parquet"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    logger.info("Found %d parquet files", len(files))
    return files


def _tokenize_single_parquet(args: tuple) -> dict:
    """
    Worker function: tokenise a single parquet file into a .bin shard.

    Runs in a separate process for true parallelism.
    Returns metadata about the processed shard.
    """
    parquet_path, output_path, shard_id, text_column = args

    import pyarrow.parquet as pq
    import tiktoken

    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    encode = enc.encode_ordinary

    try:
        table = pq.read_table(parquet_path, columns=[text_column])
    except Exception as e:
        logger.error("Failed to read %s: %s", parquet_path, e)
        return {"shard_id": shard_id, "status": "error", "error": str(e)}

    all_tokens: list[int] = []
    docs_processed = 0

    for batch in table.to_batches(max_chunksize=10_000):
        texts = batch.column(text_column).to_pylist()
        for text in texts:
            if not text or len(text.strip()) < 50:
                continue
            tokens = encode(text)
            all_tokens.extend(tokens)
            docs_processed += 1

    if not all_tokens:
        return {"shard_id": shard_id, "status": "empty", "tokens": 0}

    # Write as uint16 binary
    arr = np.array(all_tokens, dtype=DTYPE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    arr.tofile(output_path)

    del table, all_tokens, arr
    gc.collect()

    return {
        "shard_id": shard_id,
        "status": "ok",
        "tokens": len(arr) if 'arr' not in dir() else docs_processed,  # fallback
        "parquet": os.path.basename(parquet_path),
        "output": output_path,
        "docs": docs_processed,
        "bytes": os.path.getsize(output_path),
    }


def tokenize_parallel(
    input_dir: str,
    output_dir: str,
    num_workers: int = 8,
    text_column: str = "text",
) -> dict:
    """
    Tokenise all parquet files in parallel into .bin shards.

    Each parquet file → one .bin shard.
    Uses multiprocessing for true CPU parallelism.
    """
    parquet_files = find_parquet_files(input_dir)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    shard_dir = os.path.join(output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    # Check manifest for already-completed shards
    manifest_path = os.path.join(output_dir, "tokenize_manifest.json")
    completed_shards: set[str] = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
            completed_shards = set(manifest.get("completed", []))
        logger.info("Resuming: %d shards already complete", len(completed_shards))

    # Build work items (skip completed)
    work_items = []
    for i, pq_path in enumerate(parquet_files):
        shard_name = f"shard_{i:06d}.bin"
        if shard_name in completed_shards:
            continue
        out_path = os.path.join(shard_dir, shard_name)
        work_items.append((pq_path, out_path, i, text_column))

    if not work_items:
        logger.info("✅ All shards already tokenised!")
        return _load_manifest(manifest_path)

    logger.info("=" * 70)
    logger.info("🔤 TOKENISING %d parquet files → .bin shards", len(work_items))
    logger.info("   Workers: %d", num_workers)
    logger.info("   Output: %s", shard_dir)
    logger.info("   Tokenizer: tiktoken/%s", TOKENIZER_NAME)
    logger.info("=" * 70)

    start = time.time()
    total_tokens = 0
    total_docs = 0
    results = []

    # Use multiprocessing for true parallelism
    with mp.Pool(processes=num_workers) as pool:
        from tqdm import tqdm
        for result in tqdm(
            pool.imap_unordered(_tokenize_single_parquet, work_items),
            total=len(work_items),
            desc="Tokenising shards",
            unit="shard",
        ):
            results.append(result)
            if result["status"] == "ok":
                completed_shards.add(f"shard_{result['shard_id']:06d}.bin")
                total_tokens += result.get("tokens", 0)
                total_docs += result.get("docs", 0)

            # Periodic manifest save (crash-safe)
            if len(results) % 50 == 0:
                _save_manifest(manifest_path, completed_shards, total_tokens, total_docs)

    elapsed = time.time() - start

    # Final manifest
    meta = _save_manifest(manifest_path, completed_shards, total_tokens, total_docs)
    meta["elapsed_minutes"] = elapsed / 60
    meta["tokens_per_second"] = total_tokens / max(elapsed, 1)

    logger.info("=" * 70)
    logger.info("✅ Tokenisation complete!")
    logger.info("   Shards: %d", len(completed_shards))
    logger.info("   Tokens: %s (%.2f B)", f"{total_tokens:,}", total_tokens / 1e9)
    logger.info("   Docs: %s", f"{total_docs:,}")
    logger.info("   Time: %.1f minutes (%.1f hours)", elapsed / 60, elapsed / 3600)
    logger.info("   Speed: %.0f tokens/sec", meta["tokens_per_second"])
    logger.info("=" * 70)

    return meta


def _save_manifest(path: str, completed: set, tokens: int, docs: int) -> dict:
    meta = {
        "completed": sorted(completed),
        "total_shards": len(completed),
        "total_tokens": tokens,
        "total_docs": docs,
        "dtype": "uint16",
        "tokenizer": f"tiktoken/{TOKENIZER_NAME}",
        "dataset": DATASET_REPO,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Merge shards into train.bin + val.bin
# ═══════════════════════════════════════════════════════════════════════════

def merge_shards(
    output_dir: str,
    val_fraction: float = VAL_FRACTION,
) -> dict:
    """
    Merge all .bin shards into final train.bin and val.bin files.

    Splits the last `val_fraction` of shards into validation.
    """
    shard_dir = os.path.join(output_dir, "shards")
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.bin")))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    logger.info("Merging %d shards into train.bin + val.bin...", len(shard_files))

    # Split: last N shards → val, rest → train
    n_val_shards = max(1, int(len(shard_files) * val_fraction))
    train_shards = shard_files[:-n_val_shards]
    val_shards = shard_files[-n_val_shards:]

    logger.info("  Train shards: %d", len(train_shards))
    logger.info("  Val shards: %d", len(val_shards))

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    train_tokens = _concat_bins(train_shards, train_path)
    val_tokens = _concat_bins(val_shards, val_path)

    # Write metadata
    for name, path, count in [("train", train_path, train_tokens), ("val", val_path, val_tokens)]:
        meta = {
            "num_tokens": count,
            "dtype": "uint16",
            "tokenizer": f"tiktoken/{TOKENIZER_NAME}",
            "dataset": DATASET_REPO,
            "file_size_bytes": os.path.getsize(path),
        }
        meta_path = path.replace(".bin", ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    logger.info("✅ Merge complete:")
    logger.info("   train.bin: %s tokens (%.2f GB)", f"{train_tokens:,}",
                os.path.getsize(train_path) / 1e9)
    logger.info("   val.bin:   %s tokens (%.2f GB)", f"{val_tokens:,}",
                os.path.getsize(val_path) / 1e9)

    return {"train_tokens": train_tokens, "val_tokens": val_tokens}


def _concat_bins(shard_paths: list[str], output_path: str) -> int:
    """Concatenate multiple .bin files into one, streaming to avoid OOM."""
    total_tokens = 0

    with open(output_path, "wb") as out_f:
        for shard_path in shard_paths:
            data = np.fromfile(shard_path, dtype=DTYPE)
            out_f.write(data.tobytes())
            total_tokens += len(data)
            del data

    return total_tokens


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Verify
# ═══════════════════════════════════════════════════════════════════════════

def verify(data_dir: str):
    """Quick sanity check on the final .bin files."""
    for name in ["train.bin", "val.bin"]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            logger.warning("⚠️  %s not found", path)
            continue

        data = np.memmap(path, dtype=DTYPE, mode="r")
        n_tokens = len(data)
        size_gb = os.path.getsize(path) / 1e9

        # Sample check
        sample = data[:100].astype(np.int64)
        logger.info(
            "✅ %s: %s tokens (%.2f GB) | min=%d max=%d | first 10: %s",
            name, f"{n_tokens:,}", size_gb, sample.min(), sample.max(),
            sample[:10].tolist(),
        )

        # Check for obviously bad values
        if sample.max() > 60000:
            logger.warning("⚠️  Token values seem high — check tokenizer compatibility")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download & tokenise full FineWeb-Edu (1.3T tokens) at maximum speed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + tokenise + merge)
  python download_fineweb_edu.py all --output-dir ./data --num-workers 16

  # Download only (parquet files)
  python download_fineweb_edu.py download --output-dir ./data/fineweb-edu-raw

  # Tokenise already-downloaded parquet files
  python download_fineweb_edu.py tokenize --input-dir ./data/fineweb-edu-raw --output-dir ./data -w 16

  # Merge shards into train.bin + val.bin
  python download_fineweb_edu.py merge --output-dir ./data

  # Verify final files
  python download_fineweb_edu.py verify --data-dir ./data
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── download ──
    dl = sub.add_parser("download", help="Download parquet files from HuggingFace")
    dl.add_argument("--output-dir", "-o", required=True, help="Directory for raw parquet files")
    dl.add_argument("--method", choices=["cli", "api"], default="cli",
                    help="Download method (cli=fastest, api=fallback)")

    # ── tokenize ──
    tok = sub.add_parser("tokenize", help="Tokenise parquet files into .bin shards")
    tok.add_argument("--input-dir", "-i", required=True, help="Directory with parquet files")
    tok.add_argument("--output-dir", "-o", required=True, help="Directory for .bin output")
    tok.add_argument("--num-workers", "-w", type=int, default=8, help="Parallel workers")
    tok.add_argument("--text-column", default="text", help="Column name for text")

    # ── merge ──
    mg = sub.add_parser("merge", help="Merge .bin shards into train.bin + val.bin")
    mg.add_argument("--output-dir", "-o", required=True, help="Directory with shards/")
    mg.add_argument("--val-fraction", type=float, default=VAL_FRACTION)

    # ── verify ──
    vf = sub.add_parser("verify", help="Verify final .bin files")
    vf.add_argument("--data-dir", "-d", required=True, help="Directory with train.bin/val.bin")

    # ── all ──
    al = sub.add_parser("all", help="Full pipeline: download → tokenize → merge → verify")
    al.add_argument("--output-dir", "-o", required=True, help="Base output directory")
    al.add_argument("--num-workers", "-w", type=int, default=8, help="Parallel workers")

    args = parser.parse_args()

    if args.command == "download":
        if args.method == "cli":
            download_with_cli(args.output_dir)
        else:
            download_with_python_api(args.output_dir)

    elif args.command == "tokenize":
        tokenize_parallel(args.input_dir, args.output_dir, args.num_workers)

    elif args.command == "merge":
        merge_shards(args.output_dir, args.val_fraction)

    elif args.command == "verify":
        verify(args.data_dir)

    elif args.command == "all":
        raw_dir = os.path.join(args.output_dir, "fineweb-edu-raw")
        t0 = time.time()

        logger.info("🚀 FULL PIPELINE: FineWeb-Edu → train.bin + val.bin")
        logger.info("=" * 70)

        # Step 1: Download
        download_with_cli(raw_dir)

        # Step 2: Tokenise
        tokenize_parallel(raw_dir, args.output_dir, args.num_workers)

        # Step 3: Merge
        merge_shards(args.output_dir)

        # Step 4: Verify
        verify(args.output_dir)

        total_elapsed = time.time() - t0
        logger.info("=" * 70)
        logger.info("🎉 DONE! Total time: %.1f hours", total_elapsed / 3600)
        logger.info("   Your data is ready at: %s/train.bin", args.output_dir)
        logger.info("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
