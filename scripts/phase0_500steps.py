#!/usr/bin/env python
"""Phase 0: 500-step test — does 1M model break through loss 7.7 plateau?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset

device = torch.device("cuda:0")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cfg = ModelConfig(
    vocab_size=8192, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4,
    d_ff=384, max_seq_len=256,
)
model = Transformer(cfg).to(device)

# Use 2 full shards (~50M tokens) so we don't repeat too much
data_dir = Path("data/tokenized/phase1_real/")
shards = sorted(data_dir.glob("train_*.bin"))
tokens_list = [np.fromfile(str(s), dtype=np.uint16) for s in shards[:2]]
tokens = np.concatenate(tokens_list)
ds = PretrainDataset(torch.from_numpy(tokens.astype(np.int64)), seq_len=256)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
model.train()

losses = []
for step, batch in enumerate(loader):
    if step >= 500:
        break
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    _, outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())
    if step < 20 or step % 50 == 0:
        print(f"  step {step:3d} | loss={losses[-1]:.4f} | ppl={np.exp(losses[-1]):.0f}")

print(f"\nResults: {losses[0]:.2f} -> {losses[-1]:.2f} (delta={losses[0]-losses[-1]:.1f})")
print(f"Best loss: {min(losses):.4f} at step {losses.index(min(losses))}")
print(f"Min PPL: {np.exp(min(losses)):.0f}")

# Dynamic verdict
if losses[-1] < 5.0:
    print(f"\n✅ 1M model CAN learn — needs >100 steps")
    print(f"   Pipeline verified. LR schedule + step count were both bottlenecks.")
elif losses[-1] < losses[100] * 0.9 and losses[-1] < 7.0:
    print(f"\n⚠️  Slowly improving — 500 steps still not enough for 1M tiny model")
else:
    print(f"\n⚠️  Minimal improvement beyond step 100")
    print(f"   step 100-500 avg loss: {np.mean(losses[100:]):.4f}")
    print(f"   This is concerning — model may be capacity-limited")
    print(f"   Try: larger d_model / more layers")
