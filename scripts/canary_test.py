#!/usr/bin/env python
"""Canary test: train on tiny clean data and verify model CAN overfit.
If model CAN overfit → pipeline is fine, data is the problem.
If model CANNOT overfit → pipeline has an undetected bug.
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
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Canary data: 5 sentences repeated ──
canary_texts = [
    "今天天气很好，阳光明媚，适合出去散步。小鸟在树上唱歌，微风吹过，树叶沙沙作响。",
    "这个机器学习模型使用了Transformer架构，通过自注意力机制来捕捉序列中的长距离依赖关系。",
    "量子计算机利用量子比特的叠加态和纠缠态，能够在某些特定问题上超越经典计算机的计算能力。",
    "北京是中国的首都，拥有悠久的历史和丰富的文化遗产，每年吸引数百万游客前来参观。",
    "深度神经网络由多个隐藏层组成，每层通过非线性激活函数对输入进行变换以提取特征。",
]
# Repeat each 200 times → 1000 sentences → ~150K chars
repeated = canary_texts * 200
tokenizer = load_tokenizer("tokenizers/phase1_8k_real/tokenizer.json")

all_ids = []
for text in repeated:
    ids = tokenizer.encode(text).ids
    all_ids.append(1)  # BOS
    all_ids.extend(ids)
    all_ids.append(2)  # EOS

tokens = torch.tensor(all_ids, dtype=torch.long)
print(f"Canary data: {len(repeated)} sentences, {len(tokens)} tokens")
print(f"Sample decoded: {tokenizer.decode(tokens[:50].tolist(), skip_special_tokens=False)}...")

# ── Model ──
cfg = ModelConfig(vocab_size=8192, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4,
                   d_ff=384, max_seq_len=128)
model = Transformer(cfg).to(device)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

# Create fixed batch of 32 sequences x 128 tokens from the canary data
seq_len = 128
total_seqs = len(tokens) // seq_len
B = 8  # small batch

print(f"\n{'='*55}")
print(f"CANARY TEST: 500 steps, constant LR=1e-3")
print(f"{'='*55}")

losses = []
for step in range(500):
    # Random batch of sequences
    idx = torch.randint(0, max(0, total_seqs - B), (1,)).item()
    batch_ids = tokens[idx * seq_len:(idx + B) * seq_len].reshape(B, seq_len).to(device)

    _, outputs = model(batch_ids, labels=batch_ids)
    loss = outputs["loss"]

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

    losses.append(loss.item())

    if step < 20 or step % 50 == 0:
        ppl = np.exp(loss.item())
        print(f"  step {step:3d} | loss={loss.item():.4f} | ppl={ppl:7.1f}")

# ── Final ──
final_loss = losses[-1]
final_ppl = np.exp(final_loss)
print(f"\nResults: {losses[0]:.2f} -> {final_loss:.4f} (delta={losses[0]-final_loss:.1f})")
print(f"Min loss: {min(losses):.4f} at step {losses.index(min(losses))}")

if final_loss < 0.5 and final_ppl < 2.0:
    print(f"\n✅ CANARY TEST PASSED — model CAN overfit clean data")
    print(f"   Pipeline is healthy. Root cause IS the training data quality.")
elif final_loss < 1.0 and final_ppl < 3.0:
    print(f"\n⚠️  CANARY TEST MARGINAL — model learning but not overfitting fully")
    print(f"   Pipeline mostly OK. Data may be part of the problem.")
else:
    print(f"\n❌ CANARY TEST FAILED — model CANNOT overfit even on 5 repeated sentences")
    print(f"   Pipeline has a fundamental bug not caught by the 5 previous diagnostics.")
