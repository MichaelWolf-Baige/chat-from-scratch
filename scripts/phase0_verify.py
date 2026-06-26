#!/usr/bin/env python
"""Phase 0: Pipeline verification —— 1M model + 1M tokens + 100 steps.
Must pass before any LR/data optimization work.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.train.optimizer import create_optimizer, CosineWarmupScheduler

device = torch.device("cuda:0")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── 1M parameter tiny model ──
cfg = ModelConfig(
    vocab_size=8192, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4,
    d_ff=384, max_seq_len=256, rope_theta=10000.0,
)
model = Transformer(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params")
print(f"Expected initial loss: ~9.0 (ln(8192)≈9.01, natural lang ~6-7.5)")

# ── Load 1M tokens from real data ──
data_dir = Path("data/tokenized/phase1_real/")
shards = sorted(data_dir.glob("train_*.bin"))
tokens = np.fromfile(str(shards[0]), dtype=np.uint16)[:1_050_000]
tokens_tensor = torch.from_numpy(tokens.astype(np.int64))
ds = PretrainDataset(tokens_tensor, seq_len=256)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

print(f"Data: {len(ds)} samples (~{len(tokens)//1_000_000}M tokens), bs=32, seq=256")

# ── Use SAME cosine decay LR schedule (to test: will learning stop early?) ──
opt = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
sched = CosineWarmupScheduler(opt, warmup_steps=10, total_steps=100)

model.train()
losses = []
grad_norms = []

print(f"\n{'='*55}")
print(f"Phase 0: 100-step pipeline verification (cosine decay LR)")
print(f"{'='*55}")

for step, batch in enumerate(loader):
    if step >= 100:
        break

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward
    logits, outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]

    # Backward
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step(step)
    opt.zero_grad()

    loss_val = loss.item()
    losses.append(loss_val)
    grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

    if step < 10 or step % 10 == 0:
        lr = opt.param_groups[0]["lr"]
        print(f"  step {step:3d} | loss={loss_val:.4f} | lr={lr:.2e} | grad={grad_norms[-1]:.3f}")

    if np.isnan(loss_val) or np.isinf(loss_val):
        print(f"  ❌ NaN/Inf at step {step}")
        break

# ── Results ──
print(f"\n{'='*55}")
print(f"Results:")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
print(f"  Delta:        {losses[0] - losses[-1]:.4f}")
print(f"  Best loss:    {min(losses):.4f} (step {losses.index(min(losses))})")

# ── Verdict ──
delta = losses[0] - losses[-1]
if delta > 2.0 and losses[-1] < 5.0:
    print(f"\n✅ PIPELINE PASSED — loss dropped {delta:.1f} points, learning works")
    print(f"   → Problem IS LR schedule (confirmed: models CAN learn with this pipeline)")
    print(f"   → Proceed to LR range test + WSD schedule")
elif delta > 1.0:
    print(f"\n⚠️  PIPELINE MARGINAL — loss dropped {delta:.1f} but still at {losses[-1]:.1f}")
    print(f"   → May have pipeline issues AND LR issues")
    print(f"   → Check: is loss stuck after step ~20? (LR decay kicking in)")
    # Quick check: compare loss at step 20 vs step 90
    early = np.mean(losses[15:25])
    late = np.mean(losses[85:95])
    print(f"   → Avg loss steps 15-25: {early:.4f}")
    print(f"   → Avg loss steps 85-95: {late:.4f}")
    if late > early * 0.95:
        print(f"   → Loss stagnated after step 20 (LR decay too aggressive)")
    else:
        print(f"   → Loss still decreasing, might need more steps")
else:
    print(f"\n❌ PIPELINE FAILED — loss barely changed ({delta:.1f})")
    print(f"   → Likely code-level bug, check:")
    print(f"     1. attention mask correctness")
    print(f"     2. loss ignore_index for padding")
    print(f"     3. tokenizer output quality")
    print(f"     4. gradient flow through layers")
