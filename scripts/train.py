#!/usr/bin/env python
"""Main training entry point.

Supports:
    - Single-GPU training (Phase 1-2)
    - DDP multi-GPU training (Phase 2-3)
    - Resume from checkpoint

Usage:
    # Phase 1: single GPU
    python scripts/train.py --config configs/train/phase1.yaml

    # Phase 2: 4 GPUs with DDP
    torchrun --nproc_per_node=4 scripts/train.py --config configs/train/phase2.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset, make_dataloader
from src.data.tokenizer_utils import load_tokenizer
from src.train.optimizer import create_optimizer, CosineWarmupScheduler
from src.train.trainer import Trainer
from src.train.distributed import setup_distributed, cleanup_distributed, wrap_ddp, is_main_process
from src.utils.logging import Logger
from src.utils.checkpoint import load_checkpoint


def load_config(config_path: str) -> dict:
    """Load YAML config and resolve model/data config includes."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load model config
    model_cfg_path = config.get("model_config", "")
    if model_cfg_path:
        full_path = Path(config_path).parent.parent / model_cfg_path
        if full_path.exists():
            with open(full_path, "r") as f:
                config["model"] = yaml.safe_load(f)
            print(f"Loaded model config from {full_path}")

    return config


def load_data(data_path: str, seq_len: int) -> torch.Tensor:
    """Load pre-tokenized data from .bin files or directory of shards."""
    data_path = Path(data_path)

    if data_path.is_dir():
        # Load all shards
        shards = sorted(data_path.glob("train_*.bin"))
        if not shards:
            raise FileNotFoundError(f"No train_*.bin files in {data_path}")

        import numpy as np
        tokens_list = []
        for shard in shards:
            tokens_list.append(np.fromfile(shard, dtype=np.uint16))
        tokens = np.concatenate(tokens_list)
        print(f"Loaded {len(tokens):,} tokens from {len(shards)} shards")
        return torch.from_numpy(tokens.astype(np.int64))

    elif data_path.suffix == ".bin":
        import numpy as np
        tokens = np.fromfile(data_path, dtype=np.uint16)
        print(f"Loaded {len(tokens):,} tokens from {data_path}")
        return torch.from_numpy(tokens.astype(np.int64))

    else:
        raise ValueError(f"Unsupported data path: {data_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model from scratch")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training config YAML"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for DDP"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    train_cfg = config
    model_cfg_dict = config.get("model", {})

    # DDP setup
    if train_cfg.get("distributed", False):
        setup_distributed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print(f"Device: {device}")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # ── Model ────────────────────────────────────────────────────────────
    model_config = ModelConfig(
        vocab_size=model_cfg_dict.get("vocab_size", 8192),
        d_model=model_cfg_dict.get("d_model", 384),
        n_layers=model_cfg_dict.get("n_layers", 6),
        n_heads=model_cfg_dict.get("n_heads", 6),
        n_kv_heads=model_cfg_dict.get("n_kv_heads", 6),
        d_ff=model_cfg_dict.get("d_ff", 1024),
        max_seq_len=model_cfg_dict.get("max_seq_len", 2048),
        rope_theta=model_cfg_dict.get("rope_theta", 10000.0),
        dropout=model_cfg_dict.get("dropout", 0.0),
        use_flash_attention=model_cfg_dict.get("use_flash_attention", True),
        use_qk_norm=model_cfg_dict.get("use_qk_norm", False),
        use_z_loss=model_cfg_dict.get("use_z_loss", False),
    )

    model = Transformer(model_config)
    params = model_config.count_parameters()
    if is_main_process():
        print(f"\nModel: {params['total']:,} parameters")
        for k, v in params.items():
            if k != "total":
                print(f"  {k}: {v:,}")

    model = model.to(device)
    if train_cfg.get("distributed", False):
        model = wrap_ddp(model)

    # ── Data ──────────────────────────────────────────────────────────────
    if is_main_process():
        print("\nLoading data...")
    seq_len = model_cfg_dict.get("max_seq_len", 2048)

    train_data_path = train_cfg.get("train_data_path", "data/tokenized/phase1/train/")
    eval_data_path = train_cfg.get("eval_data_path", "data/tokenized/phase1/eval.bin")

    train_tokens = load_data(train_data_path, seq_len)
    train_dataset = PretrainDataset(train_tokens, seq_len=seq_len)
    train_dataloader = make_dataloader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=4,
    )

    eval_tokens = load_data(eval_data_path, seq_len)
    eval_dataset = PretrainDataset(eval_tokens, seq_len=seq_len)
    eval_dataloader = make_dataloader(
        eval_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=2,
    )

    if is_main_process():
        print(f"Train: {len(train_dataset):,} samples ({len(train_tokens):,} tokens)")
        print(f"Eval:  {len(eval_dataset):,} samples ({len(eval_tokens):,} tokens)")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = create_optimizer(
        model,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 500),
        total_steps=train_cfg.get("total_steps", 10000),
        lr_min=train_cfg.get("lr_min", 1e-6),
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    resume_path = args.resume or train_cfg.get("resume_from")
    if resume_path:
        if is_main_process():
            print(f"\nResuming from {resume_path}...")
        info = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_step = info["step"]
        start_epoch = info["epoch"]
        if is_main_process():
            print(f"  Resumed at step {start_step}, epoch {start_epoch}")

    # ── Logger ────────────────────────────────────────────────────────────
    log_dir = train_cfg.get("log_dir", "logs/phase1/")
    logger = Logger(
        log_dir=log_dir,
        use_wandb=train_cfg.get("use_wandb", False),
        wandb_run_name=f"phase1_step{start_step}",
    ) if is_main_process() else None

    if is_main_process() and logger is not None:
        logger.log_hyperparams({**model_cfg_dict, **train_cfg})

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_cfg,
        logger=logger,
    )
    trainer.global_step = start_step
    trainer.epoch = start_epoch

    # ── Training Loop ─────────────────────────────────────────────────────
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/phase1/"))
    save_every = train_cfg.get("save_every", 1000)
    eval_every = train_cfg.get("eval_every", 500)

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Starting training — {train_cfg.get('total_steps', '?')} steps")
        print(f"{'='*60}\n")

    try:
        while trainer.global_step < train_cfg.get("total_steps", 10000):
            trainer.epoch += 1
            if is_main_process():
                print(f"\n--- Epoch {trainer.epoch} ---")

            # Train
            train_metrics = trainer.train_epoch(train_dataloader)
            if is_main_process():
                print(f"  Train loss: {train_metrics['train/loss']:.4f}")

            # Eval
            eval_metrics = trainer.eval_epoch(eval_dataloader)
            if is_main_process():
                print(f"  Eval loss:  {eval_metrics['eval/loss']:.4f}")
                print(f"  Perplexity: {eval_metrics['eval/perplexity']:.1f}")

            # Save checkpoint
            if trainer.global_step % save_every == 0 and is_main_process():
                ckpt_path = checkpoint_dir / f"step_{trainer.global_step}.pt"
                trainer.save(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

            # Check if we've exceeded total steps
            if trainer.global_step >= train_cfg.get("total_steps", 10000):
                break

        # Final save
        if is_main_process():
            final_path = checkpoint_dir / "final.pt"
            trainer.save(final_path, is_best=True)
            print(f"\n{'='*60}")
            print(f"Training complete! Final checkpoint: {final_path}")
            print(f"{'='*60}")

    except KeyboardInterrupt:
        if is_main_process():
            print("\nInterrupted. Saving checkpoint...")
            trainer.save(checkpoint_dir / f"interrupt_step_{trainer.global_step}.pt")
            print("Checkpoint saved. Exiting.")

    finally:
        if logger is not None:
            logger.close()
        if train_cfg.get("distributed", False):
            cleanup_distributed()


if __name__ == "__main__":
    main()
