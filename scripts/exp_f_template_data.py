#!/usr/bin/env python
"""Experiment F: Template-generated diverse training data.

Strategy: Generate synthetic data with high token repetition from templates.
Covers multiple domains: encyclopedia, news, dialogue, code, reasoning.
Theory: if rarity is the bottleneck, templated data with controlled vocabulary
should train efficiently.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.tokenizer_utils import load_tokenizer

device = torch.device("cuda:0")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

tokenizer = load_tokenizer("tokenizers/phase1_8k_real/tokenizer.json")

# ── Step 1: Template-based data generation ──
print("Generating template-based training data...")

# Controlled vocabulary (common words only)
subjects = ["模型", "系统", "算法", "网络", "数据", "代码", "函数", "方法", "工具", "框架",
            "语言", "结构", "架构", "接口", "引擎", "平台", "服务", "应用", "模块", "组件"]
verbs_cn = ["处理", "分析", "计算", "训练", "生成", "预测", "优化", "设计", "实现", "测试"]
adjs_cn = ["高效", "稳定", "灵活", "精准", "可靠", "先进", "强大", "轻量", "智能", "快速"]

subjects_en = ["model", "system", "algorithm", "network", "data", "code", "function",
               "method", "tool", "framework", "language", "structure"]
verbs_en = ["process", "analyze", "compute", "train", "generate", "predict",
            "optimize", "design", "implement", "test"]
adjs_en = ["efficient", "robust", "flexible", "accurate", "reliable", "advanced",
           "powerful", "lightweight", "intelligent", "fast"]

# Template bank (>50 variants across 5 domains)
templates = []
# Encyclopedia style
for _ in range(15):
    templates.append(f"{random.choice(subjects)}是一种{random.choice(adjs_cn)}的{random.choice(subjects)}，用于{random.choice(verbs_cn)}和{random.choice(verbs_cn)}{random.choice(subjects)}。")
# News style
for _ in range(15):
    templates.append(f"据最新研究显示，{random.choice(adjs_cn)}的{random.choice(subjects)}能够显著提升{random.choice(subjects)}的{random.choice(subjects)}效果。")
# Dialogue style
for _ in range(10):
    templates.append(f"用户：请问{random.choice(subjects)}的主要特点是什么？\n助手：{random.choice(subjects)}的核心优势在于{random.choice(adjs_cn)}的{random.choice(subjects)}设计，可以高效地{random.choice(verbs_cn)}{random.choice(subjects)}。")
# Code style (generates Python snippets)
for _ in range(10):
    templates.append(f"def {random.choice(verbs_en)}_{random.choice(subjects_en)}(x):\n    result = x + {random.randint(1,100)}\n    return result\n\n# This function performs {random.choice(verbs_en)} on {random.choice(subjects_en)}")
# Reasoning style (chain-of-thought)
for _ in range(10):
    templates.append(f"问题：为什么{random.choice(subjects)}需要{random.choice(adjs_cn)}的{random.choice(subjects)}？\n分析：首先，{random.choice(subjects)}的核心是{random.choice(subjects)}。其次，{random.choice(adjs_cn)}的{random.choice(subjects)}能够{random.choice(verbs_cn)}。因此，{random.choice(subjects)}必须采用{random.choice(adjs_cn)}的设计。")

# Generate 5000 samples
texts = []
for i in range(5000):
    tmpl = random.choice(templates)
    texts.append(tmpl)

# Tokenize
all_ids = []
for text in texts:
    ids = tokenizer.encode(text).ids
    all_ids.append(1)
    all_ids.extend(ids)
    all_ids.append(2)

tokens = torch.tensor(all_ids, dtype=torch.long)
unique_tok = len(torch.unique(tokens))
avg_len = len(all_ids) / len(texts)
print(f"Generated {len(texts)} texts, {len(tokens):,} tokens total")
print(f"Unique token types: {unique_tok}/{8192} ({unique_tok/8192:.1%}), avg {avg_len:.0f} tok/text")
print(f"Sample: {texts[0][:100]}...")

# ── Step 2: Model + Training ──
cfg = ModelConfig.phase1()
cfg.max_seq_len = 256
model = Transformer(cfg).to(device)
print(f"Model: {cfg.total_params:,} params")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
model.train()

seq_len = 256
total_seqs = len(tokens) // seq_len
B = 16
total_steps = 1000

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
    if step <= 20 or step % 100 == 0:
        ppl = np.exp(loss.item())
        print(f"  step {step:4d} | loss={loss.item():.4f} | ppl={ppl:.0f}")

final = np.mean(losses[-20:])
ppl = np.exp(final)
print(f"\n{'='*50}")
print(f"F: Template Data | {total_steps} steps | {len(texts)} samples")
print(f"  Loss: {losses[0]:.2f} -> {final:.4f}, PPL: {np.exp(losses[0]):.0f} -> {ppl:.0f}")

if ppl < 30:
    print(f"  ✅ EXCELLENT: Template data works very well")
elif ppl < 100:
    print(f"  ✅ WORKS: Significant improvement over raw wiki")
elif ppl < 500:
    print(f"  ⚠️  MARGINAL: Better but not great")
else:
    print(f"  ❌ FAIL: Template diversity insufficient")
