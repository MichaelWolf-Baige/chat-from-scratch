"""Unified logging interface for TensorBoard and WandB."""

from __future__ import annotations

import os
from pathlib import Path


class Logger:
    """Simple logger wrapping TensorBoard (and optionally WandB).

    Usage:
        logger = Logger(log_dir="logs/phase1", use_wandb=False)
        logger.log_scalar("train/loss", 3.14, step=100)
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        use_wandb: bool = False,
        wandb_project: str = "chat-from-scratch",
        wandb_run_name: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self._tb_writer = None
        self._wandb_run = None

        # Lazy init TensorBoard
        self._tb = None

        # Lazy init WandB
        if use_wandb:
            import wandb
            self._wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                dir=str(self.log_dir),
            )

    def _ensure_tb(self):
        """Lazily create TensorBoard writer."""
        if self._tb is None:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric.

        Args:
            tag: Metric name (e.g. 'train/loss').
            value: Scalar value.
            step: Global step.
        """
        if self.use_wandb and self._wandb_run is not None:
            self._wandb_run.log({tag: value}, step=step)
        else:
            self._ensure_tb()
            self._tb.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram (e.g. gradient norms, activation values)."""
        if self.use_wandb and self._wandb_run is not None:
            self._wandb_run.log({tag: values}, step=step)
        else:
            self._ensure_tb()
            self._tb.add_histogram(tag, values, step)

    def log_hyperparams(self, params: dict) -> None:
        """Log hyperparameters."""
        if self.use_wandb and self._wandb_run is not None:
            self._wandb_run.config.update(params)

    def close(self) -> None:
        """Close loggers."""
        if self._tb is not None:
            self._tb.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
