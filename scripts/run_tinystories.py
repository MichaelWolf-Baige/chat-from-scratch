#!/usr/bin/env python
"""TinyStories benchmark — reproduce known 14M PPL ~15-25 baseline."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, numpy as np, json, time
from datetime import datetime
from src.model.config import ModelConfig
from src.model.transformer import Transformer

device = torch.device("cuda:0")
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

print("=" * 55)
print("TinyStories Benchmark: 14M Model")
print("=" * 55)

# ── Download TinyStories ──
from datasets import load_dataset
from transformers import GPT2Tokenizer

cache_path = Path("data/tiny_stories")
cache_path.mkdir(parents=True, exist_ok=True)

print("Downloading TinyStories...")
try:
    ds = load_dataset("roneneldan/TinyStories", split="train")
except Exception as e:
    print(f"  Failed: {e}")
    # Try with explicit token
    ds = load_dataset("roneneldan/TinyStories", split="train", token=True)

texts = [s["text"].strip() for s in ds]
print(f"  Loaded {len(texts):,} stories")

# ── Tokenize (GPT-2) ──
print("Tokenizing...")
tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
all_ids = []
for text in texts:
    ids = tok.encode(text)
    all_ids.append(tok.eos_token_id)
    all_ids.extend(ids)
    all_ids.append(tok.eos_token_id)
tokens = torch.tensor(all_ids, dtype=torch.long)
print(f"  Vocab: {tok.vocab_size}, Tokens: {len(tokens):,}")

# ── Train/Val split ──
seq_len = 256; bs = 32
total_seqs = len(tokens) // seq_len
usable = total_seqs * seq_len
tokens_flat = tokens[:usable].view(total_seqs, seq_len)
split = int(total_seqs * 0.95)
train_t, val_t = tokens_flat[:split], tokens_flat[split:]
print(f"  Train: {len(train_t)} seqs, Val: {len(val_t)} seqs")

# ── Model (~14M, GPT-2 vocab) ──
cfg = ModelConfig(
    vocab_size=tok.vocab_size, d_model=128, n_layers=6, n_heads=4, n_kv_heads=4,
    d_ff=384, max_seq_len=256, rope_theta=10000.0, dropout=0.0,
    use_flash_attention=True, tie_word_embeddings=True, rms_norm_eps=1e-6,
    pad_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
)
model = Transformer(cfg).to(device)
n = cfg.total_params
print(f"  Model: {n:,} params, d={cfg.d_model}, L={cfg.n_layers}, d_ff={cfg.d_ff}")
print(f"  Embedding: {cfg.count_parameters()['embedding']:,} ({cfg.count_parameters()['embedding']/n:.0%})")

# ── Train ──
epochs = 3
steps_per_epoch = len(train_t) // bs
total_steps = steps_per_epoch * epochs
lr = 5e-4
warmup = total_steps // 10
decay_start = int(total_steps * 0.85)

opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
model.train()

print(f"\n  Steps: {total_steps} ({epochs} epochs), LR: {lr} WSD")
print(f"  Tokens: {total_steps * bs * seq_len:,}")
print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")

losses = []; gs = 0; t0 = time.time()

for epoch in range(epochs):
    perm = torch.randperm(len(train_t))
    for i in range(0, len(train_t) - bs, bs):
        idx = perm[i:i+bs]; batch = train_t[idx].to(device)
        _, out = model(batch, labels=batch); loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()

        if gs < warmup:
            lr_now = lr * (gs + 1) / warmup
        elif gs < decay_start:
            lr_now = lr
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

        # Eval
        if gs % 500 == 0:
            model.eval(); et = []; en = 0
            with torch.no_grad():
                for vi in range(0, min(len(val_t)-bs, bs*10), bs):
                    vb = val_t[vi:vi+bs].to(device)
                    _, eo = model(vb, labels=vb); et.append(eo["loss"].item()); en += 1
            ep = np.exp(np.mean(et))
            print(f"  >>> VAL PPL: {ep:.0f} <<<")
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
print(f"  Model: {n:,} params | Steps: {gs} | Time: {elapsed/60:.1f}min")
print(f"  Train: loss {losses[0]:.2f}->{losses[-1]:.4f} | ppl {np.exp(losses[0]):.0f}->{np.exp(losses[-1]):.0f}")
print(f"  VAL PPL: {val_ppl:.0f}")

if val_ppl < 30:
    print(f"\n  ✅ PIPELINE HEALTHY — pipeline works, focus on Chinese data/hyperparams")
elif val_ppl < 60:
    print(f"\n  ⚠️  MARGINAL — learning but not optimally")
else:
    print(f"\n  ❌ ISSUE — fundamental problem in pipeline, fix before changing data")

# Demo
print(f"\n{'='*55}\nGeneration Demo\n{'='*55}")
prompt = "Once upon a time, there was a little"
pid = torch.tensor([tok.encode(prompt)], device=device)
with torch.no_grad():
    full_ids, _ = model.generate(pid, max_new_tokens=50, temperature=0.8, top_k=30, top_p=0.9,
                                  eos_token_id=tok.eos_token_id)
print(f"  Prompt: {prompt}")
print(f"  Story:  {tok.decode(full_ids[0].tolist(), skip_special_tokens=True)[:250]}")
print(f"\n✅ Done!")
