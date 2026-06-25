"""Evaluation metrics: perplexity, accuracy, loss."""

from __future__ import annotations

import math


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss (natural log base).

    PPL = exp(loss)

    Interpretation:
        PPL ≈ 10.4: random model (equal to vocab_size for uniform distribution
                     over 32K vocab, but varies with actual token distribution)
        PPL < 100:  model is learning patterns
        PPL < 20:   reasonable language model
        PPL < 10:   good model for general text
    """
    return math.exp(loss)


def compute_accuracy(logits, labels, ignore_index: int = -100) -> float:
    """Compute token-level prediction accuracy.

    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        ignore_index: Token id to ignore (e.g. padding).

    Returns:
        Accuracy as a float in [0, 1].
    """
    import torch

    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0

    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels) & mask
    return (correct.sum() / mask.sum()).item()
