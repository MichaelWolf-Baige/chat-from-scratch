#!/usr/bin/env python
"""Phase 1 formal training on real data with nohup support.

Usage:
    nohup python scripts/run_phase1_train.py > /tmp/phase1_train.log 2>&1 &
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import numpy as np
import json
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset, make_dataloader
from src.train.optimizer import create_optimizer, CosineWarmupScheduler
from src.utils.checkpoint import save_checkpoint


def main():
    device = torch.device("cuda:1")  # GPU 1 (GPU 0 in use by others)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Config ──
    bs, seq = 32, 2048
    lr = 3e-4
    warmup_steps = 200
    total_steps = 2000  # 2000 * 32 * 2048 ≈ 131M tokens (close to our data)
    eval_every = 200
    save_every = 500
    grad_accum = 1
    log_every = 10

    # ── Model ──
    cfg = ModelConfig.phase1()
    model = Transformer(cfg).to(device)
    model.train()
    print(f"Phase 1 Training | {cfg.total_params:,} params | {total_steps} steps")
    print(f"Batch: {bs} x {seq} = {bs*seq:,} tokens/step")
    print(f"Target: {total_steps * bs * seq:,} tokens total")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── Data ──
    data_dir = Path("data/tokenized/phase1_real/")
    shards = sorted(data_dir.glob("train_*.bin"))
    tokens_list = [np.fromfile(str(s), dtype=np.uint16) for s in shards]
    train_tokens = np.concatenate(tokens_list)
    train_tensor = torch.from_numpy(train_tokens.astype(np.int64))

    eval_data = np.fromfile(str(data_dir / "eval.bin"), dtype=np.uint16)
    eval_tensor = torch.from_numpy(eval_data.astype(np.int64))

    train_ds = PretrainDataset(train_tensor, seq_len=seq)
    eval_ds = PretrainDataset(eval_tensor, seq_len=seq)
    train_loader = make_dataloader(train_ds, batch_size=bs, shuffle=True, num_workers=0)  # num_workers=0 to avoid multiprocessing deadlock
    eval_loader = make_dataloader(eval_ds, batch_size=bs, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds):,} samples, Eval: {len(eval_ds):,} samples")

    # ── Optimizer ──
    opt = create_optimizer(model, learning_rate=lr, weight_decay=0.1)
    sched = CosineWarmupScheduler(opt, warmup_steps=warmup_steps, total_steps=total_steps)

    # ── Training ──
    ckpt_dir = Path("checkpoints/phase1_real")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    train_losses = []
    eval_losses = []
    tok_start = time.time()
    tokens_total = 0

    epoch = 0
    while global_step < total_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] / grad_accum
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(global_step)
            opt.zero_grad()

            loss_val = outputs["loss"].item()
            train_losses.append(loss_val)
            tokens_total += input_ids.numel()
            global_step += 1

            if global_step % log_every == 0:
                elapsed = time.time() - tok_start
                tps = tokens_total / elapsed
                lr_now = opt.param_groups[0]["lr"]
                grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5

                print(f"  step {global_step:5d}/{total_steps} | loss={loss_val:.4f} | "
                      f"lr={lr_now:.2e} | grad={grad_norm:.2f} | {tps/1000:.0f}K tok/s")

            if global_step % eval_every == 0:
                model.eval()
                eval_total = 0.0
                eval_count = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        ei = eval_batch["input_ids"].to(device)
                        el = eval_batch["labels"].to(device)
                        _, eout = model(ei, labels=el)
                        eval_total += eout["loss"].item()
                        eval_count += 1
                        if eval_count >= 5:  # limit eval
                            break
                eval_loss = eval_total / max(eval_count, 1)
                eval_losses.append((global_step, eval_loss))
                ppl = np.exp(eval_loss)
                print(f"  >>> eval @ step {global_step}: loss={eval_loss:.4f}, ppl={ppl:.1f} <<<")
                model.train()

            if global_step % save_every == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                save_checkpoint(ckpt_path, model, opt, sched, step=global_step, epoch=epoch, config={})
                print(f"  💾 checkpoint: {ckpt_path}")

            if np.isnan(loss_val) or np.isinf(loss_val):
                print(f"  ❌ NaN/Inf at step {global_step}! Stopping.")
                return

    # ── Final ──
    elapsed = time.time() - tok_start
    final_path = ckpt_dir / "final.pt"
    save_checkpoint(final_path, model, opt, sched, step=global_step, epoch=epoch, config={})

    model.eval()
    eval_total = 0.0
    eval_count = 0
    with torch.no_grad():
        for eval_batch in eval_loader:
            ei = eval_batch["input_ids"].to(device)
            el = eval_batch["labels"].to(device)
            _, eout = model(ei, labels=el)
            eval_total += eout["loss"].item()
            eval_count += 1
            if eval_count >= 20:
                break
    final_eval_loss = eval_total / max(eval_count, 1)
    final_ppl = np.exp(final_eval_loss)

    print(f"\n{'='*60}")
    print(f"Phase 1 Training Complete!")
    print(f"  Steps: {global_step}")
    print(f"  Tokens: {tokens_total:,}")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
    print(f"  Avg throughput: {tokens_total/elapsed:,.0f} tok/s")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final eval loss: {final_eval_loss:.4f}")
    print(f"  Final perplexity: {final_ppl:.1f}")
    print(f"  Checkpoint: {final_path}")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Save run summary ──
    summary = {
        "model_params": cfg.total_params,
        "batch_size": bs,
        "seq_len": seq,
        "total_steps": global_step,
        "total_tokens": tokens_total,
        "train_time_seconds": elapsed,
        "final_train_loss": train_losses[-1],
        "final_eval_loss": final_eval_loss,
        "final_perplexity": final_ppl,
        "train_losses": train_losses[::10],  # subsample
        "eval_steps": eval_losses,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    with open(ckpt_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {ckpt_dir / 'run_summary.json'}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
