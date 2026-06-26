#!/usr/bin/env python
"""Experiment E: Curated Wikipedia + multi-epoch.

Strategy: Select 1000 mid-length (50-300 char) Wikipedia articles.
Train 5 epochs on this fixed set. Theory: if rarity is the bottleneck,
repeated exposure to a manageable vocabulary should solve it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.tokenizer_utils import load_tokenizer

device = torch.device("cuda:0")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = load_tokenizer("tokenizers/phase1_8k_real/tokenizer.json")

# ── Step 1: Select curated documents ──
print("Selecting curated Wikipedia documents...")
docs = []
with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "")
        ln = len(text)
        # Mid-length: informative but not overwhelming
        if 50 <= ln <= 300:
            docs.append(text)
        if len(docs) >= 1000:
            break

print(f"Selected {len(docs)} docs, avg length: {np.mean([len(d) for d in docs]):.0f} chars")

# ── Step 2: Tokenize ──
all_ids = []
for text in docs:
    ids = tokenizer.encode(text).ids
    all_ids.append(1)   # BOS
    all_ids.extend(ids)
    all_ids.append(2)   # EOS

tokens = torch.tensor(all_ids, dtype=torch.long)
unique_tok = len(torch.unique(tokens))
print(f"Tokens: {len(tokens):,} total, {unique_tok} unique types "
      f"({unique_tok/8192:.1%} of vocab)")

# ── Step 3: Model ──
cfg = ModelConfig.phase1()
cfg.max_seq_len = 256
model = Transformer(cfg).to(device)
print(f"Model: {cfg.total_params:,} params")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
model.train()

seq_len = 256
total_seqs = len(tokens) // seq_len
B = 16
epochs = 5
steps_per_epoch = max(1, total_seqs // B)

print(f"\nEpochs: {epochs}, Steps/epoch: ~{steps_per_epoch}, Batch: {B}x{seq_len}")
print(f"{'Epoch':<8} {'Step':<8} {'Loss':<10} {'PPL':<10}")

losses = []
global_step = 0
for epoch in range(epochs):
    # Shuffle sequences each epoch
    perm = torch.randperm(total_seqs)
    for i in range(0, min(total_seqs - B, steps_per_epoch * B), B):
        indices = perm[i:i + B]
        batch = torch.stack([tokens[j * seq_len:(j + 1) * seq_len] for j in indices]).to(device)

        _, outputs = model(batch, labels=batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())
        global_step += 1

        if global_step <= 20 or global_step % 50 == 0:
            ppl = np.exp(loss.item())
            ep = epoch + (i / max(steps_per_epoch * B, 1))
            print(f"{ep:<8.1f} {global_step:<8} {loss.item():<10.4f} {ppl:<10.0f}")

final = np.mean(losses[-20:])
ppl = np.exp(final)
print(f"\n{'='*50}")
print(f"E: Curated Wiki | {epochs} epochs | {len(docs)} docs")
print(f"  Loss: {losses[0]:.2f} -> {final:.4f}")
print(f"  PPL:  {np.exp(losses[0]):.0f} -> {ppl:.0f}")

if ppl < 50:
    print(f"  ✅ SOLVED: Multi-epoch on curated data works")
elif ppl < 200:
    print(f"  ⚠️  PARTIAL: Better than before but still high")
else:
    print(f"  ❌ STILL STUCK at ppl={ppl:.0f}")
