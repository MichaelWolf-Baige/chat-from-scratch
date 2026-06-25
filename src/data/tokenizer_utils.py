"""Tokenizer utilities: load, save, encode, decode.

Supports HuggingFace tokenizers (trained via scripts/train_tokenizer.py).
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer


def load_tokenizer(path: str | Path) -> HFTokenizer:
    """Load a HuggingFace tokenizer from disk.

    Args:
        path: Path to tokenizer.json file or directory containing it.

    Returns:
        HuggingFace Tokenizer instance.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}")

    return HFTokenizer.from_file(str(path))


def save_tokenizer(tokenizer: HFTokenizer, path: str | Path) -> None:
    """Save a HuggingFace tokenizer to disk.

    Args:
        tokenizer: HuggingFace Tokenizer instance.
        path: Directory to save to (will create tokenizer.json inside).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path / "tokenizer.json"))


def get_token_count(text: str, tokenizer: HFTokenizer) -> int:
    """Count tokens in a text string."""
    return len(tokenizer.encode(text).ids)


def encode_batch(
    texts: list[str], tokenizer: HFTokenizer, max_length: int = 2048
) -> list[list[int]]:
    """Encode a batch of texts into token lists.

    Args:
        texts: List of text strings.
        tokenizer: HuggingFace Tokenizer.
        max_length: Maximum token length (truncates from the beginning).

    Returns:
        List of token id lists.
    """
    encodings = tokenizer.encode_batch(texts)
    return [
        enc.ids[:max_length] for enc in encodings
    ]


def decode_tokens(
    token_ids: list[int], tokenizer: HFTokenizer, skip_special_tokens: bool = True
) -> str:
    """Decode token ids back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
