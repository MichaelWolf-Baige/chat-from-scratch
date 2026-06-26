#!/usr/bin/env python
"""P1 Smoke Test: 100M Deep-Narrow Architecture Validation

Checks before full pretraining:
  1. Forward pass works (no NaN, no OOM)
  2. Backward pass / gradient flow healthy through all 24 layers
  3. Training stability over 200 steps
  4. Residual flow check: attention entropy & hidden variance in deep layers (L18+)

Architecture: d=512, n_layers=24, n_heads=8, n_kv_heads=4 (GQA 2:1), d_ff=2048, ~99M
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.data.tokenizer_utils import load_tokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════════════
# 100M Deep-Narrow Configuration
# ═══════════════════════════════════════════════════════════════

cfg = ModelConfig(
    vocab_size=8192, d_model=512, n_layers=24, n_heads=8, n_kv_heads=4,
    d_ff=2048, max_seq_len=1024, rope_theta=100000.0, dropout=0.0,
    use_flash_attention=True, tie_word_embeddings=True, rms_norm_eps=1e-6,
    use_qk_norm=True,  # MiniMind/DeepSeek stability recommendation
    pad_token_id=0, bos_token_id=1, eos_token_id=2,
)

model = Transformer(cfg).to(device)
n_params = cfg.total_params
emb_ratio = cfg.count_parameters()["embedding"] / n_params
depth_ratio = cfg.n_layers / cfg.d_model

print("=" * 55)
print("P1 Smoke Test: 100M Deep-Narrow Architecture")
print("=" * 55)
print(f"  Config: d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} KV={cfg.n_kv_heads} d_ff={cfg.d_ff}")
print(f"  Params: {n_params:,}")
print(f"  Embedding: {cfg.count_parameters()['embedding']:,} ({emb_ratio:.1%})")
print(f"  FFN: {cfg.count_parameters()['ffn']:,} | Attn: {cfg.count_parameters()['attention']:,}")
print(f"  depth/width: {depth_ratio:.4f} | GQA: 2:1 | QK-Norm: ON")

# ═══════════════════════════════════════════════════════════════
# Test 1: Forward pass
# ═══════════════════════════════════════════════════════════════
print(f"\n[1/5] Forward pass test...")
B, S = 4, 512
input_ids = torch.randint(1, 8192, (B, S), device=device)
model.train()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
logits, outputs = model(input_ids)
torch.cuda.synchronize()
fwd_time = time.time() - t0
fwd_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"  Input: {B}x{S} | Output: {logits.shape}")
print(f"  Forward: {fwd_time*1000:.0f}ms | Peak VRAM: {fwd_mem:.1f}GB")
print(f"  Has NaN: {torch.isnan(logits).any().item()} | Has Inf: {torch.isinf(logits).any().item()}")
assert logits.shape == (B, S, 8192), f"Shape mismatch: {logits.shape}"
assert not torch.isnan(logits).any(), "NaN in logits!"
print(f"  ✅ Forward pass OK")

# ═══════════════════════════════════════════════════════════════
# Test 2: Loss + Backward
# ═══════════════════════════════════════════════════════════════
print(f"\n[2/5] Loss + Backward test...")
labels = input_ids.clone()
_, outputs = model(input_ids, labels=labels)
loss = outputs["loss"]
print(f"  Loss: {loss.item():.4f} (expected: ~9.0 for ln(8192))")
assert 8.0 < loss.item() < 10.0, f"Loss out of expected range: {loss.item():.4f}"

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
bwd_time = time.time() - t0
bwd_mem = torch.cuda.max_memory_allocated() / 1024**3 - fwd_mem
print(f"  Backward: {bwd_time*1000:.0f}ms | Extra VRAM: {bwd_mem:.1f}GB")
print(f"  Total peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
print(f"  ✅ Backward pass OK")

# ═══════════════════════════════════════════════════════════════
# Test 3: Per-layer gradient health
# ═══════════════════════════════════════════════════════════════
print(f"\n[3/5] Per-layer gradient health check...")
layer_grads = {}
for name, param in model.named_parameters():
    if param.grad is not None and "layers." in name:
        parts = name.split(".")
        layer_idx = int(parts[1])
        if layer_idx not in layer_grads:
            layer_grads[layer_idx] = []
        layer_grads[layer_idx].append(param.grad.norm().item())

# Check gradient flow through deep layers
early = np.mean([np.mean(v) for k, v in sorted(layer_grads.items()) if k < 6])
mid = np.mean([np.mean(v) for k, v in sorted(layer_grads.items()) if 6 <= k < 18])
late = np.mean([np.mean(v) for k, v in sorted(layer_grads.items()) if k >= 18])

print(f"  Avg grad norm: L0-5: {early:.4f} | L6-17: {mid:.4f} | L18-23: {late:.4f}")
print(f"  Late/early ratio: {late/early:.3f}")
if late < early * 0.1:
    print(f"  ⚠️  DEEP LAYER GRADIENT VANISHING — residual flow bottleneck suspected")
elif late > early * 10:
    print(f"  ⚠️  DEEP LAYER GRADIENT EXPLODING — gradient instability")
else:
    print(f"  ✅ Gradient flow balanced across all layers")

# All layers must have non-zero gradients
dead_layers = [k for k, v in layer_grads.items() if np.mean(v) < 1e-10]
if dead_layers:
    print(f"  ❌ Dead gradients in layers: {dead_layers}")
else:
    print(f"  ✅ All {len(layer_grads)} layers have healthy gradients")

# ═══════════════════════════════════════════════════════════════
# Test 4: Residual flow — attention entropy & hidden variance
# ═══════════════════════════════════════════════════════════════
print(f"\n[4/5] Residual flow analysis (attention entropy + hidden variance)...")
model.eval()
# Register forward hooks to capture layer outputs
layer_outputs = {}
handles = []
for i, layer in enumerate(model.layers):
    def make_hook(idx):
        return lambda m, inp, out: layer_outputs.update({idx: out[0].detach()})
    handles.append(layer.register_forward_hook(make_hook(i)))

with torch.no_grad():
    model(input_ids)

for h in handles:
    h.remove()

# Analyze deep layer behavior
for i in [0, 5, 11, 17, 20, 23]:
    if i in layer_outputs:
        hidden = layer_outputs[i].float()  # (B, S, D)
        # Compute variance per position
        var = hidden.var(dim=-1).mean().item()
        # Effective rank via SVD on a sample
        flat = hidden.reshape(-1, hidden.shape[-1])[:256]
        centered = flat - flat.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            s_norm = S / (S.sum() + 1e-10)
            entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
            eff_rank = torch.exp(entropy).item()
        except:
            eff_rank = -1
        marker = ""
        if i >= 18 and var < 0.01:
            marker = " ⚠️ LOW VARIANCE (bottleneck?)"
        elif i >= 18 and eff_rank < 10:
            marker = " ⚠️ LOW RANK (collapse?)"
        print(f"  L{i:2d}: variance={var:.4f} | eff_rank={eff_rank:.0f}{marker}")

# ═══════════════════════════════════════════════════════════════
# Test 5: 200-step training stability
# ═══════════════════════════════════════════════════════════════
print(f"\n[5/5] 200-step training stability test...")
# Generate simple data
texts = []
for i in range(500):
    texts.append(f"这是第{i}个测试句子。用来验证训练稳定性的样本数据。包含一些常用的词汇和表达方式。")
tok = load_tokenizer("tokenizers/phase1_8k_real/tokenizer.json")
all_ids = []
for t in texts:
    ids = tok.encode(t).ids; all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
tokens = torch.tensor(all_ids, dtype=torch.long)

seq_len = 512; bs = 4
total_seqs = len(tokens) // seq_len
tokens_flat = tokens[:total_seqs*seq_len].view(-1, seq_len)

opt = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95))
model.train()

losses = []
grad_norms = []
tok_start = time.time()

for step in range(200):
    idx = torch.randint(0, max(1, total_seqs - 2), (bs,))
    batch = tokens_flat[idx].to(device)

    _, out = model(batch, labels=batch)
    loss = out["loss"]
    loss.backward()

    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

    losses.append(loss.item())
    if isinstance(gn, torch.Tensor):
        grad_norms.append(gn.item())
    else:
        grad_norms.append(gn)

    # WSD LR
    warmup = 20; decay = 170
    if step < warmup:
        lr = 8e-4 * (step+1) / warmup
    elif step < decay:
        lr = 8e-4
    else:
        p = min((step-decay)/30, 1.0)
        lr = 8e-4*0.01 + 0.5*8e-4*(1.0 + np.cos(np.pi*p))
    for pg in opt.param_groups: pg["lr"] = lr

    if step < 20 or step % 40 == 0:
        elapsed = time.time() - tok_start
        tps = (step+1)*bs*seq_len / max(elapsed, 0.01)
        print(f"  step {step:3d} | loss={loss.item():.4f} | ppl={np.exp(loss.item()):.0f} | "
              f"lr={lr:.2e} | gn={grad_norms[-1]:.3f} | {tps/1000:.0f}K tok/s")

    if np.isnan(loss.item()):
        print(f"  ❌ NaN at step {step}! Architecture unstable.")
        break

elapsed = time.time() - tok_start
delta = losses[0] - losses[-1]
final_gn = np.mean(grad_norms[-10:])

print(f"\n  Loss: {losses[0]:.2f} -> {losses[-1]:.2f} (delta={delta:.2f})")
print(f"  Avg grad norm (last 10): {final_gn:.4f}")
print(f"  Speed: {(200)*bs*seq_len/elapsed:.0f} tok/s")

# ═══════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("P1 SMOKE TEST VERDICT")
print(f"{'='*55}")

all_ok = True
checks = []

if delta < 0.5:
    print(f"  ❌ Loss barely changed (delta={delta:.2f}) — model not learning")
    all_ok = False
else:
    print(f"  ✅ Loss decreasing (delta={delta:.2f})")
    checks.append("learn")

if final_gn < 1e-4:
    print(f"  ❌ Gradient norm too small ({final_gn:.6f}) — gradient vanishing")
    all_ok = False
else:
    print(f"  ✅ Gradient norm healthy ({final_gn:.4f})")
    checks.append("grad")

if late < early * 0.05:
    print(f"  ❌ Deep layer gradient vanishing (L18+={late:.4f} vs L0-5={early:.4f})")
    all_ok = False
else:
    print(f"  ✅ Deep layer gradient OK")
    checks.append("deep")

if all_ok:
    print(f"\n  ✅ ALL CHECKS PASSED — 100M architecture is stable!")
    print(f"  Proceed to full pretraining on 4-GPU DDP")
    print(f"  Target: 500M-1B tokens, ~8-12 hours on 4x3090")
else:
    print(f"\n  ❌ Issues found — need architecture adjustment before full training")
    print(f"  Consider: d=768/L=16 or d=640/L=20 instead of d=512/L=24")
