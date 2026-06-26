#!/usr/bin/env python
"""Phase 0 control experiment: constant LR vs cosine decay.

Run the same 1M model with constant LR=3e-4 to isolate:
- If constant LR works → pipeline is fine, cosine decay is the bottleneck
- If constant LR also fails → pipeline has a code-level bug
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.train.optimizer import create_optimizer

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

data_dir = Path("data/tokenized/phase1_real/")
shards = sorted(data_dir.glob("train_*.bin"))
tokens = np.fromfile(str(shards[0]), dtype=np.uint16)[:1_050_000]
ds = PretrainDataset(torch.from_numpy(tokens.astype(np.int64)), seq_len=256)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

# ── Constant LR, no decay ──
opt = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
# Don't use scheduler — just constant LR

n_params = sum(p.numel() for p in model.parameters())
print(f"=== CONSTANT LR (3e-4) vs COSINE DECAY ===")
print(f"Model: {n_params:,} params | 100 steps | bs=32, seq=256")
print(f"{'Step':<6} {'Loss':<10}")
print("-" * 22)

model.train()
losses = []
for step, batch in enumerate(loader):
    if step >= 100:
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
    if step < 10 or step % 10 == 0:
        print(f"{step:<6} {loss.item():<10.4f}")

print(f"\nResults: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[0]-losses[-1]:.1f})")

# Compare with cosine decay run
cosine_final = 7.97
improvement = cosine_final / losses[-1]
print(f"Cosine decay final: {cosine_final:.2f}")
print(f"Constant LR  final: {losses[-1]:.2f}")
print(f"Constant LR is {improvement:.1f}x better")

if losses[-1] < 4.0:
    print(f"\n✅ CONSTANT LR WORKS — pipeline is fine")
    print(f"   The 2206 PPL was entirely caused by cosine decay killing LR too early")
elif losses[-1] < cosine_final:
    print(f"\n⚠️  Constant LR better but still high ({losses[-1]:.1f})")
    print(f"   May need >100 steps for 1M model")
else:
    print(f"\n❌ Something else is wrong — both schedulers fail")
