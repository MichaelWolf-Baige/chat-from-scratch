#!/usr/bin/env python
"""Phase 1 formal training with DDP (4 GPUs).

Usage:
    torchrun --nproc_per_node=4 scripts/run_phase1_ddp.py
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import time
import numpy as np
import json
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.train.optimizer import create_optimizer, CosineWarmupScheduler
from src.utils.checkpoint import save_checkpoint


def get_dataloader(tokens, batch_size, seq_len, shuffle, rank, world_size):
    """Create dataloader with distributed sampler."""
    ds = PretrainDataset(tokens, seq_len=seq_len)
    sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=shuffle,
        drop_last=True,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        num_workers=2, pin_memory=True,
        prefetch_factor=2, persistent_workers=True,
    )


def main():
    # ── DDP init ──
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ── Seed ──
    seed = 42 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Config ──
    # Per-GPU batch size. Global batch = 32 * 4 = 128
    bs, seq = 32, 2048
    lr = 3e-4 * world_size  # linear scaling
    warmup_steps = 50
    total_steps = 500  # 500 * (32*4) * 2048 ≈ 131M tokens
    eval_every = 100
    save_every = 200
    log_every = 10

    # ── Model ──
    cfg = ModelConfig.phase1()
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)
    model.train()

    if rank == 0:
        print(f"Phase 1 DDP Training | {cfg.total_params:,} params | {world_size}x RTX 3090")
        print(f"Per-GPU batch: {bs}x{seq} | Global batch: {bs*world_size}x{seq}")
        print(f"Total steps: {total_steps} | Tokens: {total_steps * bs * world_size * seq:,}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

    # ── Data ──
    # Load same data on all ranks, DistributedSampler handles partitioning
    data_dir = Path("data/tokenized/phase1_real/")
    shards = sorted(data_dir.glob("train_*.bin"))
    tokens_list = [np.fromfile(str(s), dtype=np.uint16) for s in shards]
    train_tokens = np.concatenate(tokens_list)
    train_tensor = torch.from_numpy(train_tokens.astype(np.int64))

    eval_data = np.fromfile(str(data_dir / "eval.bin"), dtype=np.uint16)
    eval_tensor = torch.from_numpy(eval_data.astype(np.int64))

    train_loader = get_dataloader(train_tensor, bs, seq, shuffle=True, rank=rank, world_size=world_size)
    eval_loader = get_dataloader(eval_tensor, bs, seq, shuffle=False, rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Train: {len(train_loader.dataset):,} samples total "
              f"({len(train_loader):,} batches/rank)")

    # ── Optimizer ──
    opt = create_optimizer(model, learning_rate=lr, weight_decay=0.1)
    sched = CosineWarmupScheduler(opt, warmup_steps=warmup_steps, total_steps=total_steps)

    # ── Training ──
    ckpt_dir = Path("checkpoints/phase1_ddp")
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    train_losses = []
    eval_losses = []
    tok_start = time.time()
    tokens_total = 0

    for epoch in range(999):  # infinite loop, break on total_steps
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            _, outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(global_step)
            opt.zero_grad()

            loss_val = loss.item()
            tokens_total += input_ids.numel() * world_size
            global_step += 1

            if rank == 0:
                train_losses.append(loss_val)

                if global_step % log_every == 0:
                    elapsed = time.time() - tok_start
                    tps = tokens_total / elapsed
                    lr_now = opt.param_groups[0]["lr"]
                    print(f"  step {global_step:4d}/{total_steps} | loss={loss_val:.4f} | "
                          f"lr={lr_now:.2e} | grad={grad_norm:.2f} | {tps/1000:.0f}K tok/s")

                if global_step % eval_every == 0:
                    model.eval()
                    eval_total = 0.0
                    eval_count = 0
                    with torch.no_grad():
                        for ei, eval_batch in enumerate(eval_loader):
                            ein = eval_batch["input_ids"].to(device)
                            ela = eval_batch["labels"].to(device)
                            _, eout = model(ein, labels=ela)
                            eval_total += eout["loss"].item()
                            eval_count += 1
                            if ei >= 5:
                                break
                    eval_loss = eval_total / max(eval_count, 1)
                    eval_losses.append((global_step, eval_loss))
                    ppl = np.exp(eval_loss)
                    print(f"  >>> eval @ step {global_step}: loss={eval_loss:.4f}, ppl={ppl:.1f} <<<")
                    model.train()

                if global_step % save_every == 0:
                    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                    save_checkpoint(ckpt_path, model.module, opt, sched,
                                    step=global_step, epoch=epoch, config={})
                    print(f"  💾 checkpoint: {ckpt_path}")

            if np.isnan(loss_val) or np.isinf(loss_val):
                if rank == 0:
                    print(f"  ❌ NaN/Inf at step {global_step}! Stopping.")
                break

        if global_step >= total_steps:
            break

    # ── Final ──
    if rank == 0:
        elapsed = time.time() - tok_start
        final_path = ckpt_dir / "final.pt"

        # Final eval
        model.eval()
        eval_total = 0.0
        eval_count = 0
        with torch.no_grad():
            for ei, eval_batch in enumerate(eval_loader):
                ein = eval_batch["input_ids"].to(device)
                ela = eval_batch["labels"].to(device)
                _, eout = model(ein, labels=ela)
                eval_total += eout["loss"].item()
                eval_count += 1
                if ei >= 20:
                    break
        final_ppl = np.exp(eval_total / max(eval_count, 1))

        save_checkpoint(final_path, model.module, opt, sched,
                        step=global_step, epoch=0, config={})

        print(f"\n{'='*60}")
        print(f"Phase 1 DDP Training Complete!")
        print(f"  GPUs: {world_size}")
        print(f"  Steps: {global_step}")
        print(f"  Tokens: {tokens_total:,}")
        print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
        print(f"  Throughput: {tokens_total/elapsed:,.0f} tok/s (global)")
        print(f"  Final train loss: {train_losses[-1]:.4f}")
        print(f"  Final perplexity: {final_ppl:.1f}")
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

        summary = {
            "gpus": world_size,
            "model_params": cfg.total_params,
            "batch_size_per_gpu": bs,
            "global_batch_size": bs * world_size,
            "total_steps": global_step,
            "total_tokens": tokens_total,
            "train_time_seconds": elapsed,
            "throughput_tok_per_sec": tokens_total / elapsed,
            "train_losses_sample": train_losses[::10],
            "eval_losses": eval_losses,
            "final_perplexity": final_ppl,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        with open(ckpt_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n✅ Done! Summary: {ckpt_dir / 'run_summary.json'}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
