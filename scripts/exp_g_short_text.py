#!/usr/bin/env python
"""Experiment G: High-frequency short texts.

Strategy: Extract first paragraph + section headings from Wikipedia.
Short texts = less rare vocabulary per sample = more repetition.
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

# ── Step 1: Extract short, high-density texts ──
print("Extracting short texts from Wikipedia...")
texts = []
with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "")
        # Strategy: take first 2-3 sentences per article
        # This gives high-frequency intro content, skipping rare details
        sentences = text.replace("\n", "。").split("。")
        intro = "。".join(sentences[:3])  # first 3 sentences
        intro = intro.strip()
        if 30 <= len(intro) <= 200:  # short, focused
            texts.append(intro)
        if len(texts) >= 5000:
            break

print(f"Extracted {len(texts)} short texts, avg length: {np.mean([len(t) for t in texts]):.0f} chars")

# ── Step 2: Tokenize ──
all_ids = []
for text in texts:
    ids = tokenizer.encode(text).ids
    all_ids.append(1)
    all_ids.extend(ids)
    all_ids.append(2)

tokens = torch.tensor(all_ids, dtype=torch.long)
unique_tok = len(torch.unique(tokens))
print(f"Tokens: {len(tokens):,} total, {unique_tok} unique types ({unique_tok/8192:.1%} of vocab)")

# ── Step 3: Model (1M for fast test, then 14M) ──
for model_label, cfg_builder in [("1M", lambda: ModelConfig(
    vocab_size=8192, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4,
    d_ff=384, max_seq_len=128
)), ("14M", ModelConfig.phase1)]:

    cfg = cfg_builder()
    if hasattr(cfg, 'max_seq_len'):
        pass  # phase1 sets it
    if cfg.max_seq_len > 256:
        cfg.max_seq_len = 256

    model = Transformer(cfg).to(device)
    n = cfg.total_params
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    model.train()

    seq_len = min(cfg.max_seq_len, 128)
    total_seqs = len(tokens) // seq_len
    B = 16
    total_steps = 500

    losses = []
    for step in range(total_steps):
        idx = torch.randint(0, max(1, total_seqs - B), (1,)).item()
        batch = tokens[idx * seq_len:(idx + B) * seq_len].reshape(B, seq_len).to(device)
        _, outputs = model(batch, labels=batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    final = np.mean(losses[-20:])
    ppl = np.exp(final)
    print(f"  {model_label:<6} | loss: {losses[0]:.2f} -> {final:.4f} | ppl: {ppl:.0f}")

    # If 1M test fails, 14M likely also fails
    if model_label == "1M" and ppl > 200:
        print(f"  ⚠️  1M already struggling — skipping 14M (same data)")
        break

    del model, opt
    torch.cuda.empty_cache()

print(f"\n{'='*50}")
print(f"G: Short Text | {len(texts)} samples | high-frequency intro sentences")
