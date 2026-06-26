#!/usr/bin/env python
"""Diagnose why loss plateaus at 7.70 across all configurations.

Checks in priority order:
1. Gradient underflow (fp16 precision)
2. Data encoding corruption
3. Loss computation correctness
4. Attention mask direction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.data.tokenizer_utils import load_tokenizer

device = torch.device("cuda:0")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Load data and model ──
cfg = ModelConfig(vocab_size=8192, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4,
                   d_ff=384, max_seq_len=256)

data_dir = Path("data/tokenized/phase1_real/")
tokens_raw = np.fromfile(str(sorted(data_dir.glob("train_*.bin"))[0]), dtype=np.uint16)[:100_000]
ds = PretrainDataset(torch.from_numpy(tokens_raw.astype(np.int64)), seq_len=256)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

batch = next(iter(loader))
input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

print("=" * 60)
print("LOSS PLATEAU DIAGNOSIS")
print("=" * 60)

# ── Test 1: Data encoding sanity ──
print("\n[1] DATA ENCODING CHECK")
tok = load_tokenizer("tokenizers/phase1_8k_real/tokenizer.json")
# Decode first sample
decoded = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)
print(f"  First 200 decoded chars: {repr(decoded[:200])}")
has_garbled = any(0x80 <= ord(c) < 0x100 for c in decoded[:200]) and not any(
    '一' <= c <= '鿿' for c in decoded[:200]
)
if has_garbled:
    print("  ❌ DATA MAY BE CORRUPTED (non-UTF-8, non-CJK characters)")
else:
    print("  ✅ Data encoding looks normal")

# ── Test 2: Check raw token distribution ──
print("\n[2] TOKEN DISTRIBUTION CHECK")
flat_tokens = input_ids.flatten().cpu().numpy()
unique, counts = np.unique(flat_tokens, return_counts=True)
vocab_used = len(unique)
print(f"  Unique tokens in batch: {vocab_used}/{cfg.vocab_size} ({vocab_used/cfg.vocab_size:.1%})")
print(f"  Most common token IDs: {unique[np.argsort(-counts)][:10]}")

# Check special tokens
pad_id = cfg.pad_token_id
bos_id = cfg.bos_token_id
eos_id = cfg.eos_token_id
unk_id = 3  # <unk>
pad_pct = (flat_tokens == pad_id).mean()
bos_pct = (flat_tokens == bos_id).mean()
eos_pct = (flat_tokens == eos_id).mean()
unk_pct = (flat_tokens == unk_id).mean()
print(f"  PAD(0): {pad_pct:.1%}, BOS(1): {bos_pct:.1%}, EOS(2): {eos_pct:.1%}, UNK(3): {unk_pct:.1%}")

if unk_pct > 0.05:
    print(f"  ❌ UNK rate {unk_pct:.1%} is too high (>5%)")
elif pad_pct > 0.3:
    print(f"  ⚠️  PAD rate {pad_pct:.1%} is high")

# ── Test 3: Gradient flow ──
print("\n[3] GRADIENT FLOW CHECK")
model = Transformer(cfg).to(device)
model.train()

# Test with fp32 first (control)
model_fp32 = Transformer(cfg).to(device)
model_fp32.train()

print("  === fp32 control ===")
_, outputs_fp32 = model_fp32(input_ids, labels=labels)
loss_fp32 = outputs_fp32["loss"]
loss_fp32.backward()

total_norm = 0.0
layer_norms = []
for name, param in model_fp32.named_parameters():
    if param.grad is not None and "weight" in name:
        n = param.grad.norm().item() ** 2
        total_norm += n
        layer_norms.append((name, n ** 0.5))

total_norm = total_norm ** 0.5
print(f"    loss={loss_fp32.item():.4f}, total_grad_norm={total_norm:.4f}")
print(f"    Per-layer grad norms:")
for name, n in layer_norms[:6]:
    print(f"      {name}: {n:.6f}")

# Test with AMP (autocast)
print("\n  === fp16 AMP (autocast) ===")
model_amp = Transformer(cfg).to(device)
model_amp.train()

with torch.amp.autocast("cuda", dtype=torch.float16):
    _, outputs_amp = model_amp(input_ids, labels=labels)
    loss_amp = outputs_amp["loss"]

loss_amp.backward()

total_norm_amp = 0.0
zero_grad_count = 0
for name, param in model_amp.named_parameters():
    if param.grad is not None and "weight" in name:
        n = param.grad.norm().item()
        total_norm_amp += n ** 2
        if n < 1e-8:
            zero_grad_count += 1

total_norm_amp = total_norm_amp ** 0.5
print(f"    loss={loss_amp.item():.4f}, total_grad_norm={total_norm_amp:.6f}")
print(f"    Zero-gradient params: {zero_grad_count}")
print(f"    fp32/fp16 grad ratio: {total_norm / max(total_norm_amp, 1e-10):.1f}x")

# Test with bf16
print("\n  === bf16 AMP (autocast) ===")
model_bf16 = Transformer(cfg).to(device)
model_bf16.train()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _, outputs_bf16 = model_bf16(input_ids, labels=labels)
    loss_bf16 = outputs_bf16["loss"]
loss_bf16.backward()

total_norm_bf16 = 0.0
zero_grad_bf16 = 0
for name, param in model_bf16.named_parameters():
    if param.grad is not None and "weight" in name:
        n = param.grad.norm().item()
        total_norm_bf16 += n ** 2
        if n < 1e-8:
            zero_grad_bf16 += 1

total_norm_bf16 = total_norm_bf16 ** 0.5
print(f"    loss={loss_bf16.item():.4f}, total_grad_norm={total_norm_bf16:.6f}")
print(f"    Zero-gradient params: {zero_grad_bf16}")

# ── Test 4: Loss computation validation ──
print("\n[4] LOSS COMPUTATION CHECK")
# Manual loss computation
with torch.no_grad():
    logits_fp32 = model_fp32(input_ids)[0]  # (B, S, V)
    shift_logits = logits_fp32[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Model's loss (uses ignore_index=pad_token_id=0)
    loss_model = F.cross_entropy(
        shift_logits.reshape(-1, cfg.vocab_size),
        shift_labels.reshape(-1),
        ignore_index=cfg.pad_token_id,
    )
    print(f"  Model loss (ignore_index=0):  {loss_model.item():.4f}")

    # Expected: ignore_index=-100
    loss_correct = F.cross_entropy(
        shift_logits.reshape(-1, cfg.vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    print(f"  Correct loss (ignore_index=-100): {loss_correct.item():.4f}")

    # How many labels == -100 vs 0?
    n_neg100 = (shift_labels == -100).sum().item()
    n_zero = (shift_labels == 0).sum().item()
    n_total = shift_labels.numel()
    print(f"  Labels == -100: {n_neg100}/{n_total} ({n_neg100/n_total:.1%})")
    print(f"  Labels == 0:    {n_zero}/{n_total} ({n_zero/n_total:.1%})")

# ── Test 5: Causal mask direction ──
print("\n[5] CAUSAL MASK CHECK")
S = input_ids.shape[1]
model_check = Transformer(cfg).to(device)
model_check.eval()
with torch.no_grad():
    # Forward with explicit causal mask
    causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    logits_check, _ = model_check(input_ids, attention_mask=causal_mask)
    # Forward without mask (uses default causal)
    logits_no_mask, _ = model_check(input_ids)
    diff = (logits_check - logits_no_mask).abs().max().item()
    print(f"  Max diff between explicit/implicit mask: {diff:.10f}")
    if diff > 1e-3:
        print(f"  ⚠️  Mask behavior inconsistent!")
    else:
        print(f"  ✅ Causal mask behavior consistent")

# ── Summary ──
print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)

# Primary verdict
if total_norm_amp < 1e-3 and zero_grad_count > 10:
    print("🔴 PRIMARY CAUSE: fp16 gradient underflow")
    print("   → Gradients are being truncated to zero in fp16")
    print("   → Solution: Use bf16 autocast + ensure loss scaling")
elif has_garbled:
    print("🔴 PRIMARY CAUSE: Data encoding corruption")
    print("   → Training data contains garbled/non-readable text")
    print("   → Solution: Re-download and verify UTF-8 encoding")
else:
    print("🟡 No single obvious cause found")
    print("   → Check the individual test results above")

print(f"\nRecommended action:")
if total_norm_bf16 > total_norm_amp * 10:
    print("  → Switch to bf16 autocast for all training")
elif total_norm_amp < 1e-3:
    print("  → Add GradScaler + switch to bf16")
else:
    print("  → Focus on data quality and loss computation")
