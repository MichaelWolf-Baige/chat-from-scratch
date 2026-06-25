#!/usr/bin/env python
"""Minimal 10-step smoke test for Phase 1 pipeline verification.

Runs a tiny training loop to verify data→forward→backward→loss descent→checkpoint→generate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import numpy as np

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset, make_dataloader
from src.train.optimizer import create_optimizer, CosineWarmupScheduler
from src.utils.checkpoint import save_checkpoint, load_checkpoint


def main():
    device = torch.device("cuda:0")
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Model ──
    cfg = ModelConfig.phase1()
    model = Transformer(cfg).to(device)
    print(f"Model: {cfg.total_params:,} params, vocab={cfg.vocab_size}")

    # ── Data ──
    data_dir = Path("data/tokenized/phase1_synthetic/")
    shards = sorted(data_dir.glob("train_*.bin"))
    tokens_list = [np.fromfile(str(s), dtype=np.uint16) for s in shards]
    all_tokens = np.concatenate(tokens_list)

    # Use only enough tokens for 10 steps
    bs, seq = 16, 128  # small for smoke test
    total_needed = bs * seq * 12  # 10 steps + margin
    tokens = torch.from_numpy(all_tokens[:total_needed].astype(np.int64))

    ds = PretrainDataset(tokens, seq_len=seq)
    loader = make_dataloader(ds, batch_size=bs, shuffle=False, num_workers=0)
    print(f"Data: {len(tokens):,} tokens, {len(ds)} samples, bs={bs}, seq={seq}")

    # ── Optimizer ──
    opt = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    sched = CosineWarmupScheduler(opt, warmup_steps=3, total_steps=10)

    # ── Training ──
    model.train()
    loss_history = []
    torch.cuda.synchronize()
    t0 = time.time()

    for step, batch in enumerate(loader):
        if step >= 10:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        _, outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(step)

        loss_val = loss.item()
        loss_history.append(loss_val)
        lr = opt.param_groups[0]["lr"]

        print(f"  Step {step:3d} | loss={loss_val:.4f} | lr={lr:.2e} | "
              f"grad_norm={sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5:.2f}")

        if np.isnan(loss_val) or np.isinf(loss_val):
            print("  ❌ NaN/Inf detected — ABORT")
            break

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tokens_processed = 10 * bs * seq
    tps = tokens_processed / elapsed

    # ── Results ──
    print(f"\n{'='*50}")
    print(f"Smoke Test Results:")
    print(f"  Steps: {len(loss_history)}")
    print(f"  Loss:  {loss_history[0]:.4f} → {loss_history[-1]:.4f} "
          f"(delta: {loss_history[0] - loss_history[-1]:.4f})")
    print(f"  Speed: {tps:.0f} tok/s ({elapsed:.1f}s total)")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")

    # Verify loss decreased
    assert loss_history[-1] < loss_history[0], \
        f"Loss did not decrease! {loss_history[0]:.4f} → {loss_history[-1]:.4f}"
    assert not any(np.isnan(v) or np.isinf(v) for v in loss_history), \
        "NaN/Inf in loss history"

    print(f"  ✅ Loss decreased (no NaN/Inf)")

    # ── Checkpoint save + load ──
    ckpt_dir = Path("checkpoints/smoke_test")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "step_5.pt"
    save_checkpoint(ckpt_path, model, opt, sched, step=5, epoch=0, config={})

    # Verify load
    info = load_checkpoint(ckpt_path, model, opt, sched)
    assert info["step"] == 5, f"Step mismatch: {info['step']} != 5"

    # Verify model still works after load
    model.eval()
    test_batch = next(iter(loader))
    with torch.no_grad():
        _, out_after = model(test_batch["input_ids"].to(device), labels=test_batch["labels"].to(device))
    print(f"  ✅ Checkpoint save+load OK (step={info['step']}, loss: {out_after['loss'].item():.4f})")

    # ── Generation ──
    model.eval()
    prompt_ids = torch.tensor([[1]], device=device)  # BOS token
    with torch.no_grad():
        full, new = model.generate(prompt_ids, max_new_tokens=20, temperature=0.8)
    print(f"  ✅ Generated {new.shape[1]} tokens (shape: {full.shape})")

    print(f"\n{'='*50}")
    print(f"ALL SMOKE TESTS PASSED ✅")
    print(f"{'='*50}")

    return loss_history


if __name__ == "__main__":
    main()
