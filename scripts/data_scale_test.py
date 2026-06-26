#!/usr/bin/env python
"""Data scaling test: find the data size where model starts to plateau.

Tests: 10 docs → 100 → 1000 → 10000 → 100000 (all Wikipedia)
Same 1M model, constant LR, 200 steps each.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.tokenizer_utils import load_tokenizer

device = torch.device("cuda:0")
tokenizer = load_tokenizer("tokenizers/phase1_8k_real/tokenizer.json")

# Read raw Wikipedia docs
import json
docs = []
with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "")
        if len(text) >= 50:
            docs.append(text)
        if len(docs) >= 50000:
            break
print(f"Loaded {len(docs):,} documents")

# Tokenize docs
def tokenize_docs(doc_list):
    all_ids = []
    for text in doc_list:
        ids = tokenizer.encode(text).ids
        all_ids.append(1)  # BOS
        all_ids.extend(ids)
        all_ids.append(2)  # EOS
    return torch.tensor(all_ids, dtype=torch.long)

seq_len = 128
B = 16

print(f"{'Docs':<10} {'Tokens':<12} {'Init loss':<12} {'Final loss':<12} {'PPL':<10} {'Verdict'}")
print("-" * 75)

for n_docs in [10, 100, 500, 1000, 5000, 10000, 50000]:
    subset = docs[:n_docs]
    tokens = tokenize_docs(subset)
    n_tokens = len(tokens)
    n_seqs = n_tokens // seq_len

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    cfg = ModelConfig(vocab_size=8192, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4,
                       d_ff=384, max_seq_len=128)
    model = Transformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    model.train()

    losses = []
    for step in range(200):
        idx = torch.randint(0, max(1, n_seqs - B), (1,)).item()
        batch = tokens[idx * seq_len:(idx + B) * seq_len].reshape(B, seq_len).to(device)
        _, outputs = model(batch, labels=batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    final = np.mean(losses[-20:])  # avg last 20
    ppl = np.exp(final)

    if ppl < 10:
        v = "✅ EASILY LEARNS"
    elif ppl < 50:
        v = "⚠️  LEARNING"
    elif ppl < 200:
        v = "🟡 PLATEAUING"
    else:
        v = "🔴 STUCK"

    print(f"{n_docs:<10} {n_tokens:<12,} {losses[0]:<12.4f} {final:<12.4f} {ppl:<10.0f} {v}")

    del model, opt
    torch.cuda.empty_cache()

print(f"\nThe smallest doc count where PPL stays > 50 is the capacity limit")
print(f"If PPL is > 50 even at 1000 docs: check tokenizer/infra")
print(f"If PPL scales with doc count: it's a model capacity issue (14M too small)")
