#!/usr/bin/env python
"""Train a BPE tokenizer using the HuggingFace tokenizers library.

Usage:
    python scripts/train_tokenizer.py --data_dir data/raw/ --output tokenizers/phase1_bpe_8k/ --vocab_size 8192

Input:
    Directory of .txt or .jsonl files.

Output:
    A tokenizer.json file in the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def read_texts(data_dir: Path, max_files: int = 100) -> list[str]:
    """Read text files from a directory. Supports .txt and .jsonl."""
    texts = []
    files = sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.jsonl"))
    if max_files > 0:
        files = files[:max_files]

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if file_path.suffix == ".jsonl":
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                    except json.JSONDecodeError:
                        continue
                else:
                    text = line

                if len(text) >= 50:  # Skip very short texts
                    texts.append(text)

                # Limit total texts for tokenizer training
                if len(texts) >= 500_000:
                    return texts
    return texts


def train_tokenizer(
    data_dir: str | Path,
    output_dir: str | Path,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    max_files: int = 100,
) -> Tokenizer:
    """Train a BPE tokenizer.

    Args:
        data_dir: Directory containing text files.
        output_dir: Where to save tokenizer.json.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency.
        max_files: Max files to read (0 = all).

    Returns:
        Trained tokenizer.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading texts from {data_dir}...")
    texts = read_texts(data_dir, max_files)
    print(f"Read {len(texts):,} texts, total chars: {sum(len(t) for t in texts):,}")

    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Normalizer: Unicode normalization
    tokenizer.normalizer = Sequence([NFKC()])

    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = ByteLevelDecoder()

    # Trainer
    special_tokens = [
        "<pad>",          # 0: padding
        "<s>",            # 1: BOS
        "</s>",           # 2: EOS
        "<unk>",          # 3: unknown
        "<|system|>",     # 4: system message marker
        "<|user|>",       # 5: user message marker
        "<|assistant|>",  # 6: assistant message marker
        "<|tool_call|>",  # 7: tool call (reserved)
        "<|tool_resp|>",  # 8: tool response (reserved)
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train
    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Verify
    encoded = tokenizer.encode("你好，这是一个测试。Hello world!")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Test encoding: '{encoded.tokens}'")
    print(f"Test token IDs: {encoded.ids}")

    # Save
    save_path = output_dir / "tokenizer.json"
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to {save_path}")

    # Save special tokens mapping for reference
    vocab = tokenizer.get_vocab()
    special_map = {tok: vocab[tok] for tok in special_tokens if tok in vocab}
    with open(output_dir / "special_tokens.json", "w") as f:
        json.dump(special_map, f, indent=2, ensure_ascii=False)
    print(f"Special tokens saved to {output_dir / 'special_tokens.json'}")

    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with text files"
    )
    parser.add_argument(
        "--output", type=str, default="tokenizers/phase1_bpe_8k/",
        help="Output directory"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=8192, help="Vocabulary size"
    )
    parser.add_argument(
        "--min_frequency", type=int, default=2, help="Minimum token frequency"
    )
    parser.add_argument(
        "--max_files", type=int, default=100, help="Max files to read (0=all)"
    )

    args = parser.parse_args()

    train_tokenizer(
        data_dir=args.data_dir,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
