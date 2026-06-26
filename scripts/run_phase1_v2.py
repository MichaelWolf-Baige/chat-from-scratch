#!/usr/bin/env python
"""Phase 1 v2: Fix LR schedule (WSD) + train 14M model on 4x3090.

Key changes from v1:
1. WSD (Warmup-Stable-Decay) instead of cosine decay
2. Longer training: 2000 steps @ global_bs=128 = 524M tokens
3. LR range test built in: first 0-50 steps warmup, 50-1500 stable, 1500-2000 decay
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
import numpy as np
import json
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.train.optimizer import create_optimizer
from src.utils.checkpoint import save_checkpoint


class WSDScheduler:
    """Warmup-Stable-Decay LR schedule.

    ┌───────┐
    │       ╲
    │  stable ╲
    │  ┌─────┐ ╲____
    │  │     │      ╲___
    └──┴─────┴──────────┴→
      warmup  stable    decay
    """
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup = warmup_steps
        self.stable = stable_steps
        self.decay = decay_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total = warmup_steps + stable_steps + decay_steps

    def get_lr(self, step):
        if step < self.warmup:
            return self.max_lr * (step + 1) / max(self.warmup, 1)
        if step < self.warmup + self.stable:
            return self.max_lr
        # Cosine decay from max_lr to min_lr
        progress = (step - self.warmup - self.stable) / max(self.decay, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(np.pi * progress))

    def step(self, step):
        lr = self.get_lr(step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


def get_loader(tokens, bs, seq, shuffle, rank, world_size):
    ds = PretrainDataset(tokens, seq_len=seq)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                  shuffle=shuffle, drop_last=True)
    return torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler,
                                        num_workers=2, pin_memory=True,
                                        prefetch_factor=2, persistent_workers=True)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Config ──
    bs, seq = 32, 2048
    total_steps = 1200
    warmup_steps = 100       # 8% — gentle warmup
    stable_steps = 900       # 75% — main learning phase
    decay_steps = 200        # 17% — fine convergence
    max_lr = 8e-4            # conservative for 14M model
    min_lr = max_lr * 0.01  # 8e-6
    eval_every = 200
    save_every = 400

    tokens_total = total_steps * bs * world_size * seq

    # ── Model ──
    cfg = ModelConfig.phase1()
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                find_unused_parameters=False)
    model.train()

    if rank == 0:
        n = cfg.total_params
        print(f"Phase 1 v2 | {n:,} params | {world_size}x RTX 3090 | WSD schedule")
        print(f"Global batch: {bs*world_size}x{seq} | {tokens_total/1e9:.2f}B tokens")
        print(f"LR: warmup={warmup_steps} stable={stable_steps} decay={decay_steps}")
        print(f"Max LR: {max_lr} | Min LR: {min_lr}")
        print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

    # ── Data ──
    data_dir = Path("data/tokenized/phase1_real/")
    shards = sorted(data_dir.glob("train_*.bin"))
    tokens_list = [np.fromfile(str(s), dtype=np.uint16) for s in shards]
    train_t = torch.from_numpy(np.concatenate(tokens_list).astype(np.int64))
    eval_t = torch.from_numpy(np.fromfile(str(data_dir / "eval.bin"), dtype=np.uint16).astype(np.int64))

    train_loader = get_loader(train_t, bs, seq, True, rank, world_size)
    eval_loader = get_loader(eval_t, bs, seq, False, rank, world_size)

    if rank == 0:
        print(f"Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches/rank)")

    # ── Optimizer ──
    opt = create_optimizer(model, learning_rate=max_lr, weight_decay=0.1)
    sched = WSDScheduler(opt, warmup_steps, stable_steps, decay_steps, max_lr, min_lr)

    # ── Training ──
    ckpt_dir = Path("checkpoints/phase1_v2")
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    train_losses = []
    tok_start = time.time()
    tok_total_count = 0

    for epoch in range(999):
        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            if global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            _, outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(global_step)
            opt.zero_grad()

            loss_val = loss.item()
            tok_total_count += input_ids.numel() * world_size
            global_step += 1

            if rank == 0:
                train_losses.append(loss_val)

                if global_step < 20 or global_step % 20 == 0:
                    elapsed = time.time() - tok_start
                    tps = tok_total_count / max(elapsed, 0.01)
                    lr_now = opt.param_groups[0]["lr"]
                    ppl_now = np.exp(loss_val)
                    print(f"  step {global_step:4d}/{total_steps} | loss={loss_val:.4f} | "
                          f"ppl={ppl_now:.0f} | lr={lr_now:.2e} | {tps/1000:.0f}K tok/s")

                if global_step % eval_every == 0:
                    model.eval()
                    eval_total, eval_n = 0.0, 0
                    with torch.no_grad():
                        for ei, eb in enumerate(eval_loader):
                            if ei >= 10:
                                break
                            ein = eb["input_ids"].to(device)
                            ela = eb["labels"].to(device)
                            _, eo = model(ein, labels=ela)
                            eval_total += eo["loss"].item()
                            eval_n += 1
                    eval_loss = eval_total / max(eval_n, 1)
                    eval_ppl = np.exp(eval_loss)
                    print(f"  >>> eval @ {global_step}: loss={eval_loss:.4f} ppl={eval_ppl:.0f} <<<")
                    model.train()

                if global_step % save_every == 0:
                    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                    save_checkpoint(ckpt_path, model.module, opt, sched,
                                    step=global_step, epoch=epoch, config={"lr_schedule": "WSD"})
                    print(f"  💾 checkpoint: step_{global_step}.pt")

            if np.isnan(loss_val):
                if rank == 0:
                    print(f"  ❌ NaN at step {global_step}")
                break

        if global_step >= total_steps:
            break

    # ── Final ──
    if rank == 0:
        elapsed = time.time() - tok_start

        model.eval()
        eval_total, eval_n = 0.0, 0
        with torch.no_grad():
            for ei, eb in enumerate(eval_loader):
                if ei >= 30:
                    break
                ein = eb["input_ids"].to(device)
                ela = eb["labels"].to(device)
                _, eo = model(ein, labels=ela)
                eval_total += eo["loss"].item()
                eval_n += 1
        final_ppl = np.exp(eval_total / max(eval_n, 1))

        final_path = ckpt_dir / "final.pt"
        save_checkpoint(final_path, model.module, opt, sched,
                        step=global_step, epoch=0, config={})

        print(f"\n{'='*60}")
        print(f"Phase 1 v2 Complete!")
        print(f"  GPUs: {world_size} | Steps: {global_step} | Tokens: {tok_total_count:,}")
        print(f"  Time: {elapsed/60:.1f}min | Speed: {tok_total_count/elapsed:,.0f} tok/s")
        print(f"  Train loss: {train_losses[-1]:.4f} (ppl={np.exp(train_losses[-1]):.0f})")
        print(f"  Eval PPL: {final_ppl:.0f}")

        summary = {
            "lr_schedule": "WSD",
            "warmup": warmup_steps,
            "stable": stable_steps,
            "decay": decay_steps,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "gpus": world_size,
            "total_steps": global_step,
            "total_tokens": tok_total_count,
            "time_s": elapsed,
            "throughput": tok_total_count / elapsed,
            "train_losses_sample": train_losses[::50],  # every 50 steps
            "final_eval_ppl": final_ppl,
        }
        with open(ckpt_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Summary: {ckpt_dir / 'run_summary.json'}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
