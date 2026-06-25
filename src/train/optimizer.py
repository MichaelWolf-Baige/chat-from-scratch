"""Optimizer factory: AdamW with cosine schedule + warmup."""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.optim as optim


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.AdamW:
    """Create AdamW optimizer with weight decay only on matmul params.

    Llama convention: no weight decay on biases, norms, and embeddings.
    """

    def _no_decay(param_name: str) -> bool:
        return any(x in param_name for x in ("bias", "norm", "embed_tokens"))

    params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not _no_decay(n)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and _no_decay(n)],
            "weight_decay": 0.0,
        },
    ]

    return optim.AdamW(params, lr=learning_rate, betas=betas, eps=eps, fused=False)


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup.

    Steps:
        0 → warmup_steps:  LR increases linearly from 0 to lr_max.
        warmup_steps → end: LR decays via cosine to lr_min.

    Usage:
        scheduler = CosineWarmupScheduler(...)
        for step in range(total_steps):
            lr = scheduler.get_lr(step)
            optimizer.param_groups[0]['lr'] = lr
            ...
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def get_lr(self, step: int) -> float:
        """Compute LR for the given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * (step + 1) / max(self.warmup_steps, 1)

        if step >= self.total_steps:
            return self.lr_min

        # Cosine decay
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        return self.lr_min + 0.5 * (self.base_lrs[0] - self.lr_min) * (
            1.0 + math.cos(math.pi * progress)
        )

    def step(self, step: int) -> None:
        """Set optimizer LR for the given step."""
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
