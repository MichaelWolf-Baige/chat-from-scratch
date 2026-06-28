#!/usr/bin/env python
"""第1步诊断：最小smoke test，只验证模型+tokenizer+训练loop能否跑通。
不需要外部数据文件，直接生成假数据。
目标是快速定位"跑崩"的根因，3分钟内出结果。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, numpy as np, time

# 1. 检查tokenizer是否存在
TOK_PATH = Path("saved_models/tokenizers/phase1_8k_real_tokenizer.json")
TOK_PATH2 = Path("tokenizers/phase1_8k_real/tokenizer.json")

print("=" * 55)
print("DIAGNOSTIC SMOKE TEST")
print("=" * 55)

# Tokenizer check
if TOK_PATH.exists():
    tok_path = str(TOK_PATH)
    print(f"[OK] Tokenizer found: {tok_path}")
elif TOK_PATH2.exists():
    tok_path = str(TOK_PATH2)
    print(f"[OK] Tokenizer found: {tok_path}")
else:
    print("[FATAL] Tokenizer not found at either:")
    print(f"  - {TOK_PATH.resolve()}")
    print(f"  - {TOK_PATH2.resolve()}")
    print("  Fix: check where your tokenizer.json actually lives")
    sys.exit(1)

from tokenizers import Tokenizer
tok = Tokenizer.from_file(tok_path)
print(f"  Vocab size: {tok.get_vocab_size()}")

# 2. GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# 3. Model forward pass
from src.model.config import ModelConfig
from src.model.transformer import Transformer

cfg = ModelConfig(
    vocab_size=8192, d_model=512, n_layers=24, n_heads=8, n_kv_heads=4,
    d_ff=2048, max_seq_len=1024, rope_theta=100000.0, dropout=0.0,
    use_flash_attention=True, tie_word_embeddings=True, rms_norm_eps=1e-6,
    use_qk_norm=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
)
print(f"\nModel: {cfg.total_params:,} params | d={cfg.d_model} L={cfg.n_layers}")

model = Transformer(cfg).to(device)
model.train()

# 4. Generate fake data (in-memory, no file needed)
print("\n[Test 1] Generating fake tokenized data...")
fake_text = "人工智能是当前科技发展的前沿方向，深度学习通过神经网络模拟人脑的工作方式。"
all_ids = []
for i in range(100):
    ids = tok.encode(fake_text).ids
    all_ids.extend(ids)
tokens = torch.tensor(all_ids, dtype=torch.long)
print(f"  Tokens: {len(tokens)}")

# 5. Train 100 steps with tiny data
B, S = 4, 256
total_seqs = len(tokens) // S
tokens_flat = tokens[:total_seqs * S].view(-1, S)
print(f"  Sequences: {total_seqs} x {S}")

opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
losses, grad_norms = [], []

print(f"\n[Test 2] Training 100 steps...")
t0 = time.time()
try:
    for step in range(100):
        idx = torch.randint(0, max(1, total_seqs - B), (B,))
        batch = tokens_flat[idx].to(device)
        _, out = model(batch, labels=batch)
        loss = out["loss"]
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())
        grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

        if np.isnan(loss.item()):
            print(f"  [FAIL] NaN at step {step}!")
            break

        if step <= 5 or step % 20 == 0:
            print(f"  step {step:3d} | loss={loss.item():.4f} | grad_norm={grad_norms[-1]:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Done. {elapsed:.1f}s | {100 * B * S / elapsed:.0f} tok/s")
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[0]-losses[-1]:.2f})")
    print(f"  Grad norms: mean={np.mean(grad_norms):.4f} min={np.min(grad_norms):.4f} max={np.max(grad_norms):.4f}")

    if abs(losses[0] - losses[-1]) < 0.1:
        print("  [WARN] Loss barely moved - model not learning (maybe data issue)")
    elif losses[-1] < losses[0]:
        print("  [OK] Loss is decreasing - model is learning")
    else:
        print("  [WARN] Loss increasing - learning rate may be too high")

    # 6. Quick generation test
    print(f"\n[Test 3] Generation test...")
    model.eval()
    prompt = "人工智能"
    ids = tok.encode(prompt).ids
    pid = torch.tensor([[1] + ids], device=device)
    out_tokens = []
    for tid, is_done in model.generate_stream(pid, max_new_tokens=10, temperature=0.8, top_k=35, top_p=0.9, eos_token_id=2):
        out_tokens.append(tid)
        if is_done:
            break
    resp = tok.decode(out_tokens, skip_special_tokens=True)
    print(f"  '{prompt}' -> '{resp[:80]}'")
    print(f"  [OK] Generation works")

    print(f"\n{'=' * 55}")
    print("VERDICT: Model code is healthy, training loop works.")
    print("If you're seeing crashes, the problem is likely:")
    print("  1. Missing data files (e.g. data/distill_merged.jsonl)")
    print("  2. Wrong tokenizer path in training scripts")
    print("  3. Multi-GPU DDP issues (try single-GPU first)")
    print(f"{'=' * 55}")

except Exception as e:
    print(f"\n[FAIL] CRASHED at step {len(losses)}")
    print(f"  Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n{'=' * 55}")
    print("VERDICT: Code crash detected. Fix the error above.")
    print(f"{'=' * 55}")
