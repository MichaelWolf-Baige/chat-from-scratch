"""Checkpoint save and load.

Saves model weights, optimizer state, scheduler state, and training metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    config: dict[str, Any],
) -> None:
    """Save a full training checkpoint.

    Args:
        path: File path (e.g. 'checkpoints/step_1000.pt').
        model: The model (unwrap DDP if needed).
        optimizer: Optimizer with state.
        scheduler: LR scheduler (CosineWarmupScheduler).
        step: Global step.
        epoch: Current epoch.
        config: Training configuration dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if present
    model_state = model.state_dict() if not hasattr(model, "module") else model.module.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config,
    }

    # Save scheduler state if it has state dict
    if hasattr(scheduler, "state_dict"):
        checkpoint["scheduler"] = scheduler.state_dict()
    else:
        checkpoint["scheduler_step"] = scheduler.__dict__.get("_step", step)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str = "cuda",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        map_location: Device to load tensors to.

    Returns:
        dict with 'step', 'epoch', 'config' from the checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Load model weights
    model_state = checkpoint["model"]
    # Handle DDP wrapping
    if hasattr(model, "module"):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None:
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        elif hasattr(scheduler, "__setattr__") and "scheduler_step" in checkpoint:
            scheduler.__dict__["_step"] = checkpoint["scheduler_step"]

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "config": checkpoint.get("config", {}),
    }


def list_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """List all checkpoints sorted by step number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
    )
    return checkpoints


def cleanup_checkpoints(
    checkpoint_dir: str | Path, keep_best: int = 5, keep_every: int = 5000
) -> None:
    """Remove old checkpoints, keeping the best N and one every K steps.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        keep_best: Keep the N most recent checkpoints.
        keep_every: Keep one checkpoint every K steps.
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    if len(checkpoints) <= keep_best:
        return

    # Keep: most recent `keep_best` + one every `keep_every` steps
    to_keep = set()
    for cp in checkpoints[-keep_best:]:
        to_keep.add(cp)

    for cp in checkpoints:
        step = int(cp.stem.split("_")[-1]) if cp.stem.split("_")[-1].isdigit() else 0
        if step % keep_every == 0:
            to_keep.add(cp)

    for cp in checkpoints:
        if cp not in to_keep:
            cp.unlink()
