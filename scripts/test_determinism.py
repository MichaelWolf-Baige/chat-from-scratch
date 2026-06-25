#!/usr/bin/env python
"""Determinism test: run twice with seed=42, assert loss curves match exactly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset, make_dataloader
from src.train.optimizer import create_optimizer


def run_training(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    cfg = ModelConfig.phase1()
    model = Transformer(cfg).to(device)
    opt = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)

    data_dir = Path("data/tokenized/phase1_synthetic/")
    shards = sorted(data_dir.glob("train_*.bin"))
    tokens_list = [np.fromfile(str(s), dtype=np.uint16) for s in shards]
    all_tokens = np.concatenate(tokens_list)
    tokens = torch.from_numpy(all_tokens[:16*128*12].astype(np.int64))
    ds = PretrainDataset(tokens, seq_len=128)
    loader = make_dataloader(ds, batch_size=16, shuffle=False, num_workers=0)

    model.train()
    losses = []
    for step, batch in enumerate(loader):
        if step >= 10:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(outputs["loss"].item())

    return losses


def main():
    print("=== Determinism Test ===")
    print("Run 1 (seed=42)...")
    losses1 = run_training(42)
    print("Run 2 (seed=42)...")
    losses2 = run_training(42)

    print(f"\nStep | Run1     | Run2     | Diff")
    print("-" * 42)
    all_ok = True
    for i, (l1, l2) in enumerate(zip(losses1, losses2)):
        diff = abs(l1 - l2)
        ok = diff < 1e-5
        if not ok:
            all_ok = False
        print(f"  {i:2d} | {l1:.6f} | {l2:.6f} | {diff:.2e} {'OK' if ok else 'FAIL'}")

    print()
    if all_ok:
        print("✅ DETERMINISM VERIFIED — all losses match within 1e-5")
    else:
        print("❌ DETERMINISM FAILED — differences > 1e-5 detected")
        print("   Check: DataLoader shuffle seed, cudnn settings, dropout")


if __name__ == "__main__":
    main()
