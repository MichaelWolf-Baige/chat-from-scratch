"""Training loop with checkpointing, logging, and evaluation."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.amp

from src.train.optimizer import CosineWarmupScheduler
from src.train.distributed import is_main_process
from src.utils.logging import Logger
from src.utils.checkpoint import save_checkpoint
from src.eval.metrics import compute_perplexity


class Trainer:
    """Single-GPU / DDP training loop.

    Usage:
        trainer = Trainer(model, optimizer, scheduler, config)
        for epoch in range(epochs):
            trainer.train_epoch(dataloader)
            trainer.eval_epoch(eval_dataloader)
            trainer.save_checkpoint()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWarmupScheduler,
        config: dict[str, Any],
        logger: Logger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger

        self.device = next(model.parameters()).device
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Mixed precision
        self.use_amp = config.get("mixed_precision", "none") != "none"
        self.amp_dtype = (
            torch.bfloat16
            if config.get("mixed_precision") == "bf16"
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16)
        )

        # Gradient accumulation
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Gradient clipping
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Logging
        self.log_every = config.get("log_every", 10)
        self.eval_every = config.get("eval_every", 500)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        """Train one epoch. Returns average metrics."""
        self.model.train()
        total_loss = 0.0
        total_z_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.amp.autocast(
                "cuda", enabled=self.use_amp, dtype=self.amp_dtype
            ):
                _, outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / self.gradient_accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (num_batches + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Update LR
                self.scheduler.step(self.global_step)

                # Metrics
                batch_loss = outputs.get("loss", loss * self.gradient_accumulation_steps).item()
                batch_z_loss = outputs.get("z_loss", torch.tensor(0.0)).item()
                total_loss += batch_loss
                total_z_loss += batch_z_loss
                num_batches += 1
                self.global_step += 1

                # Log
                if self.global_step % self.log_every == 0 and is_main_process():
                    elapsed = time.time() - start_time
                    tokens_per_sec = (
                        input_ids.numel()
                        * (self.log_every * self.gradient_accumulation_steps)
                        / elapsed
                    )
                    lr = self.optimizer.param_groups[0]["lr"]
                    self._log(
                        {
                            "train/loss": batch_loss,
                            "train/z_loss": batch_z_loss,
                            "train/grad_norm": grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm,
                            "train/lr": lr,
                            "train/tokens_per_sec": tokens_per_sec,
                            "step": self.global_step,
                        }
                    )
                    start_time = time.time()

                # Evaluation
                if self.global_step % self.eval_every == 0 and is_main_process():
                    # During training eval not typical; use train loss for now
                    pass
            else:
                num_batches += 1  # count accumulation steps (shared count for clarity)

        # Fix num_batches to reflect optimizer steps
        effective_batches = max(1, num_batches // self.gradient_accumulation_steps)
        avg_loss = total_loss / effective_batches if effective_batches > 0 else 0.0
        return {"train/loss": avg_loss, "train/z_loss": total_z_loss / effective_batches}

    @torch.no_grad()
    def eval_epoch(
        self, dataloader: torch.utils.data.DataLoader
    ) -> dict[str, float]:
        """Evaluate on held-out data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.amp.autocast(
                "cuda", enabled=self.use_amp, dtype=self.amp_dtype
            ):
                _, outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        ppl = compute_perplexity(avg_loss)

        metrics = {"eval/loss": avg_loss, "eval/perplexity": ppl}
        if is_main_process():
            self._log(metrics)
        return metrics

    def save(self, path: str | Path, is_best: bool = False) -> None:
        """Save checkpoint."""
        if not is_main_process():
            return
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.epoch,
            config=self.config,
        )
        if is_best:
            self.best_eval_loss = float("inf")  # will be updated by caller

    def _log(self, metrics: dict[str, Any]) -> None:
        """Log metrics to the configured logger."""
        if self.logger is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.log_scalar(key, value, self.global_step)
