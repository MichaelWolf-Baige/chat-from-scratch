#!/usr/bin/env python
"""TinyStories fast benchmark — 200K stories, batch tokenize, verify pipeline."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, numpy as np, time, json
from datetime import datetime
from src.model.config import ModelConfig
from src.model.transformer import Transformer

device = torch.device("cuda:0")
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

print("=" * 55)
print("TinyStories Fast Benchmark")
print("=" * 55)

# ── Download & sample 200K stories ──
from datasets import load_dataset
from transformers import GPT2Tokenizer

ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
# Shuffle with seed then take N
ds = ds.shuffle(seed=42, buffer_size=10000)

N = 200_000
print(f"Sampling {N:,} stories...")
texts = []
for i, s in enumerate(ds):
    texts.append(s["text"].strip())
    if i % 20000 == 0:
        print(f"  {i}/{N}...", end="\r")
    if len(texts) >= N:
        break
print(f"\n  Sampled {len(texts):,} stories")

# ── Tokenize (batch mode for speed) ──
tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
vocab_size = tok.vocab_size
print(f"  GPT-2 vocab: {vocab_size}")

# Batch encode: ~5000 stories at a time to avoid memory blowup
all_ids = []
batch_size = 5000
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    encodings = tok(batch, add_special_tokens=True, truncation=True,
                     max_length=512, return_attention_mask=False,
                     return_token_type_ids=False)
    for ids in encodings["input_ids"]:
        all_ids.extend(ids)
    if i % 100000 == 0 and i > 0:
        print(f"  Tokenized {i}/{len(texts):,}...")

tokens = torch.tensor(all_ids, dtype=torch.long)
print(f"  Total tokens: {len(tokens):,}")

# ── Train/Val ──
seq_len = 256; bs = 32
total_seqs = len(tokens) // seq_len
usable = total_seqs * seq_len
tokens_flat = tokens[:usable].view(total_seqs, seq_len)
split = int(total_seqs * 0.95)
train_t, val_t = tokens_flat[:split], tokens_flat[split:]

# ── Model: 14M params with GPT-2 vocab ──
# Embedding: 50257 * 288 = 14.5M... too much.
# Use d_model=144: embedding = 50257*144 = 7.2M
# Per layer: 4*144*144 + 3*144*384 = 83K + 166K = 249K
# 8 layers = 2.0M → total ~9.2M
# Try d_model=192: embedding = 9.6M, per layer 4*192*192+3*192*512 = 147K+295K=442K
# 7 layers = 3.1M → total ~12.7M ✓
cfg = ModelConfig(
    vocab_size=vocab_size, d_model=192, n_layers=7, n_heads=6, n_kv_heads=6,
    d_ff=512, max_seq_len=256, rope_theta=10000.0, dropout=0.0,
    use_flash_attention=True, tie_word_embeddings=True, rms_norm_eps=1e-6,
    pad_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id,
    eos_token_id=tok.eos_token_id,
)
model = Transformer(cfg).to(device)
n = cfg.total_params
m = cfg.count_parameters()
print(f"  Model: {n:,} params | emb={m['embedding']:,} ({m['embedding']/n:.0%})")
print(f"  d={cfg.d_model} L={cfg.n_layers} heads={cfg.n_heads} d_ff={cfg.d_ff}")

# ── Train ──
epochs = 5
steps_per_epoch = len(train_t) // bs
total_steps = steps_per_epoch * epochs
lr = 5e-4
warmup = total_steps // 10
decay_start = int(total_steps * 0.85)

opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
model.train()

tokens_total = total_steps * bs * seq_len
print(f"\n  Training: {total_steps} steps | {tokens_total:,} tokens | LR={lr} WSD")
print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")

losses, gs = [], 0
t0 = time.time()

for epoch in range(epochs):
    perm = torch.randperm(len(train_t))
    for i in range(0, len(train_t) - bs, bs):
        idx = perm[i:i+bs]; batch = train_t[idx].to(device)
        _, out = model(batch, labels=batch); loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()

        if gs < warmup: lr_now = lr * (gs + 1) / warmup
        elif gs < decay_start: lr_now = lr
        else:
            p = min((gs - decay_start) / max(total_steps - decay_start, 1), 1.0)
            lr_now = lr * 0.01 + 0.5 * lr * (1.0 + np.cos(np.pi * p))
        for pg in opt.param_groups: pg["lr"] = lr_now

        losses.append(loss.item()); gs += 1

        if gs <= 20 or gs % 200 == 0:
            elapsed = time.time() - t0
            print(f"  step {gs:5d}/{total_steps} | loss={loss.item():.4f} "
                  f"ppl={np.exp(loss.item()):.0f} | lr={lr_now:.2e} | "
                  f"{gs*bs*seq_len/elapsed:.0f} tok/s")

        if gs % 500 == 0:
            model.eval(); et = []
            with torch.no_grad():
                for vi in range(0, min(len(val_t)-bs, bs*10), bs):
                    vb = val_t[vi:vi+bs].to(device)
                    _, eo = model(vb, labels=vb); et.append(eo["loss"].item())
            print(f"  >>> VAL PPL: {np.exp(np.mean(et)):.0f} <<<")
            model.train()

        if gs >= total_steps: break
    if gs >= total_steps: break

# ── Final ──
model.eval(); et = []
with torch.no_grad():
    for vi in range(0, min(len(val_t)-bs, bs*15), bs):
        vb = val_t[vi:vi+bs].to(device)
        _, eo = model(vb, labels=vb); et.append(eo["loss"].item())
val_ppl = np.exp(np.mean(et))
elapsed = time.time() - t0

print(f"\n{'='*55}")
print(f"TinyStories Results")
print(f"{'='*55}")
print(f"  Model: {n:,} params | Steps: {gs} | Tokens: {gs*bs*seq_len:,}")
print(f"  Time: {elapsed/60:.1f}min | Speed: {gs*bs*seq_len/elapsed:.0f} tok/s")
print(f"  Train: loss {losses[0]:.2f}->{losses[-1]:.4f} | ppl {np.exp(losses[0]):.0f}->{np.exp(losses[-1]):.0f}")
print(f"  VAL PPL: {val_ppl:.0f}")

print()
if val_ppl < 30:
    print("  ✅ PIPELINE HEALTHY — PPL within TinyStories expected range (15-25)")
    print("     Pipeline works. Chinese PPL 2200 is a data/hyperparam issue.")
elif val_ppl < 60:
    print("  ⚠️  MARGINAL — learning but above TinyStories baseline")
    print("     Pipeline mostly OK but there may be tuning issues.")
else:
    print("  ❌ PIPELINE ISSUE — far from TinyStories baseline")
    print("     Need to debug pipeline before working on Chinese data.")

# Generate demo
print(f"\n{'='*55}\nGeneration Demo\n{'='*55}")
prompt = "Once upon a time, there was a little"
pid = torch.tensor([tok.encode(prompt)], device=device)
with torch.no_grad():
    full_ids, _ = model.generate(pid, max_new_tokens=50, temperature=0.8, top_k=30, top_p=0.9,
                                  eos_token_id=tok.eos_token_id)
print(f"  Prompt: {prompt}")
print(f"  Story:  {tok.decode(full_ids[0].tolist(), skip_special_tokens=True)[:250]}")
print(f"\n✅ Done!")
