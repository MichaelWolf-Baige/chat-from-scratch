#!/usr/bin/env python
"""Preprocess raw text data into tokenized .bin files for training.

Usage:
    python scripts/preprocess_data.py \
        --input data/raw/ \
        --output data/tokenized/phase1/ \
        --tokenizer tokenizers/phase1_bpe_8k/tokenizer.json \
        --seq_len 2048 \
        --num_shards 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.data.tokenizer_utils import load_tokenizer


def preprocess_file(
    file_path: Path,
    tokenizer,
    seq_len: int,
    min_text_len: int = 100,
) -> list[int]:
    """Tokenize a single file into a flat list of token IDs.

    Documents are concatenated with <s>...</s> boundaries.
    Very short documents are skipped.
    """
    token_ids = []
    bos_id = tokenizer.token_to_id("<s>") or 1
    eos_id = tokenizer.token_to_id("</s>") or 2

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse JSONL or plain text
            if file_path.suffix == ".jsonl":
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    continue
            else:
                text = line

            if not text or len(text) < min_text_len:
                continue

            # Encode and wrap with BOS/EOS
            ids = tokenizer.encode(text).ids
            token_ids.append(bos_id)
            token_ids.extend(ids)
            token_ids.append(eos_id)

    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess text data for training")
    parser.add_argument(
        "--input", type=str, required=True, help="Directory with raw text files"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for .bin files"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer.json"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048, help="Sequence length"
    )
    parser.add_argument(
        "--min_text_len", type=int, default=100, help="Minimum text length (chars)"
    )
    parser.add_argument(
        "--num_shards", type=int, default=10, help="Number of output shards"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.99, help="Train split ratio"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded: vocab_size={vocab_size}")

    # Check dtype: uint16 for vocab ≤ 65535, uint32 otherwise
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    print(f"Using dtype={dtype.__name__} (vocab size {vocab_size})")

    # Find input files
    input_files = sorted(input_dir.glob("*.jsonl")) + sorted(input_dir.glob("*.txt"))
    if not input_files:
        print(f"ERROR: No .jsonl or .txt files found in {input_dir}")
        return

    print(f"Found {len(input_files)} input files")

    # Tokenize all files
    all_tokens = []
    for file_path in tqdm(input_files, desc="Tokenizing"):
        tokens = preprocess_file(file_path, tokenizer, args.seq_len, args.min_text_len)
        all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Estimated sequences: {total_tokens // args.seq_len:,}")

    # Convert to numpy
    tokens_array = np.array(all_tokens, dtype=dtype)

    # Shuffle at document level (approximate with chunk shuffle)
    # For simplicity, we skip full global shuffle here — the DataLoader handles it
    rng = np.random.default_rng(42)
    rng.shuffle(tokens_array)

    # Split into train/eval
    split_idx = int(len(tokens_array) * args.train_ratio)
    train_tokens = tokens_array[:split_idx]
    eval_tokens = tokens_array[split_idx:]

    # Write sharded train files
    train_shard_size = len(train_tokens) // args.num_shards
    for i in range(args.num_shards):
        start = i * train_shard_size
        end = start + train_shard_size if i < args.num_shards - 1 else len(train_tokens)
        shard_path = output_dir / f"train_{i:04d}.bin"
        train_tokens[start:end].tofile(shard_path)

    # Write eval file (single shard)
    eval_path = output_dir / "eval.bin"
    eval_tokens.tofile(eval_path)

    # Save metadata
    metadata = {
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "eval_tokens": len(eval_tokens),
        "vocab_size": vocab_size,
        "seq_len": args.seq_len,
        "dtype": dtype.__name__,
        "num_train_shards": args.num_shards,
        "hash_sha256": "TBD",  # Compute if available
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to {output_dir}")
    print(f"  Train: {len(train_tokens):,} tokens across {args.num_shards} shards")
    print(f"  Eval:  {len(eval_tokens):,} tokens")
    print(f"  Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
