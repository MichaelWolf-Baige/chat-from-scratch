#!/usr/bin/env python
"""Phase 0: LR range test — find optimal learning rate for 1M model."""
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

data_dir = Path("data/tokenized/phase1_real/")
shards = sorted(data_dir.glob("train_*.bin"))
tokens = np.fromfile(str(shards[0]), dtype=np.uint16)[:2_100_000]
ds = PretrainDataset(torch.from_numpy(tokens.astype(np.int64)), seq_len=256)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

print(f"{'LR':<10} {'Init Loss':<12} {'Final Loss':<12} {'Delta':<10} {'PPL':<10}")
print("-" * 55)

for lr in [3e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
    model = Transformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
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
    ppl = np.exp(losses[-1])
    print(f"{lr:<10.0e} {losses[0]:<12.4f} {losses[-1]:<12.4f} "
          f"{losses[0]-losses[-1]:<10.4f} {ppl:<10.0f}")
    if np.isnan(losses[-1]):
        print(f"  -> NaN at this LR, stop")
        break
