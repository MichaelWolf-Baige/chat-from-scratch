#!/usr/bin/env python
"""Phase 1 Two-Stage: Template foundation → Wikipedia fine-tune.

Stage 1: Pure template (28K samples, ~500 unique tokens, 10 epochs)
         → Model learns clean language patterns.
Stage 2: Add 1500 curated wiki samples (~2500 more unique tokens, 3 epochs)
         → Model integrates real vocabulary without collapse.

Total budget: ~1.5M tokens, ~15min on RTX 3090.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random
import json
import time
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.tokenizer_utils import load_tokenizer
from src.utils.checkpoint import save_checkpoint

device = torch.device("cuda:0")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = load_tokenizer("tokenizers/phase1_8k_real/tokenizer.json")

# ═══════════════════════════════════════════════════════
# Template engine (larger pool than exp_f, for quality)
# ═══════════════════════════════════════════════════════
random.seed(42)
cn_n = ["模型","算法","网络","数据","系统","代码","函数","模块","架构",
        "接口","引擎","平台","服务","框架","协议","索引","线程","进程",
        "实验","理论","变量","公式","参数","样本","信号","方法","策略",
        "方案","标准","流程","机制","结构","层次","市场","金融","物流",
        "细胞","基因","蛋白质","分子","原子","电子","磁场","温度","压力",
        "能源","气候","城市","交通","人口","经济","投资","渠道","消费者"]
cn_v = ["处理","分析","计算","训练","生成","预测","优化","设计","实现",
        "测试","评估","调整","监控","部署","扩展","集成","转换","提取",
        "检测","识别","提升","降低","加速","简化","增强","改进","推动"]
cn_a = ["高效","稳定","灵活","精准","可靠","先进","强大","智能","快速",
        "安全","完善","成熟","主流","创新","实用","专业","全面","显著"]
en_n = ["model","algorithm","network","system","data","code","function",
        "module","architecture","interface","engine","platform","framework"]
en_v = ["process","analyze","compute","train","generate","predict",
        "optimize","design","implement","test","evaluate","deploy"]
en_a = ["efficient","robust","flexible","accurate","reliable","advanced",
        "powerful","lightweight","intelligent","fast","secure","scalable"]
yrs = list(range(2018,2027)); nms = list(range(1,100))
countries = ["中国","美国","日本","德国","法国","英国","韩国"]
def p(L): return random.choice(L)

def gen_templates(n):
    texts = []
    for _ in range(n):
        r = random.random()
        s, s2 = p(cn_n), p(cn_n)
        if r < 0.20:
            texts.append(f"{s}是一种{p(cn_a)}的{s2}技术，用于{p(cn_v)}和{p(cn_v)}，已广泛应用于{p(cn_n)}领域。研究表明，{p(cn_a)}的{s}能够显著提升{p(cn_n)}的{p(cn_n)}效率。")
        elif r < 0.40:
            texts.append(f"在{p(cn_n)}领域，{p(cn_a)}的{s}起到关键作用。通过{p(cn_v)}{p(cn_n)}，可以{p(cn_v)}{s}的核心{p(cn_n)}，实现{p(cn_a)}的效果。目前已在全国多个{p(cn_n)}应用。")
        elif r < 0.55:
            texts.append(f"用户：请介绍一下{s}的主要特点和应用场景。\n助手：{s}是一种{p(cn_a)}的{s2}方案。主要特点包括：第一，{p(cn_a)}的{p(cn_n)}设计；第二，高效的{p(cn_v)}能力；第三，完善的{p(cn_n)}机制。应用场景涵盖{p(cn_n)}、{p(cn_n)}和{p(cn_n)}等领域。\n用户：它的核心原理是什么？\n助手：{s}基于{p(cn_a)}的{p(cn_n)}原理，通过{p(cn_v)}和{p(cn_v)}来处理{p(cn_n)}，从而{p(cn_v)}整体{p(cn_n)}的效率。")
        elif r < 0.68:
            texts.append(f"问题：为什么{p(cn_a)}的{s}对{s2}很重要？\n分析：第一步，{s}通过{p(cn_v)}和{p(cn_v)}来处理{p(cn_n)}。第二步，{s2}的主要瓶颈在于{p(cn_n)}效率。第三步，{p(cn_a)}的{s}恰好能{p(cn_v)}这个瓶颈。结论：{s}是提升{s2}性能的{p(cn_a)}策略。")
        elif r < 0.80:
            se, s2e = p(en_n), p(en_n)
            texts.append(f"The {se} is a {p(en_a)} {s2e} approach for {p(en_v)}ing {p(en_n)}. First proposed in {p(yrs)}, it has become a standard tool in {p(en_n)} research. Key advantages include {p(en_a)} {p(en_n)} and efficient {p(en_v)}ing.")
        elif r < 0.90:
            se, s2e = p(en_n), p(en_n)
            texts.append(f"Q: How does {se} improve {s2e} performance?\nA: {se} uses {p(en_a)} {p(en_n)} to {p(en_v)} {s2e}, resulting in {p(en_a)} accuracy on {p(en_n)} benchmarks. The key insight is that {se} can {p(en_v)} the underlying {p(en_n)} more effectively than traditional methods.")
        else:
            se = p(en_n)
            texts.append(f"def {p(en_v)}_{se}(data, threshold={p(nms)}):\n    results = []\n    for item in data:\n        score = compute_{se}_score(item)\n        if score > threshold:\n            results.append(item)\n    return sorted(results, key=lambda x: x.score, reverse=True)\n\nclass {se.capitalize()}Processor:\n    def __init__(self, config=None):\n        self.config = config or {{'lr': 0.001, 'batch': 32}}\n\n    def process(self, inputs):\n        return [{p(en_v)}(x) for x in inputs]")
    return texts

def load_wiki(n):
    """Load curated wiki paragraphs (clean, mid-length)."""
    texts = []
    with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            if 80 <= len(text) <= 350:
                texts.append(text)
            if len(texts) >= n:
                break
    return texts

def tokenize_texts(texts):
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    tokens = torch.tensor(all_ids, dtype=torch.long)
    return tokens, len(torch.unique(tokens))

def train_stage(model, opt, tokens, seq_len, bs, epochs, lr, label, device):
    """Train one stage. Returns losses list."""
    total_seqs = len(tokens) // seq_len
    steps_per_epoch = total_seqs // bs
    total_steps = steps_per_epoch * epochs
    warmup = total_steps // 10
    decay_start = int(total_steps * 0.85)

    indices = list(range(total_seqs))
    random.shuffle(indices)
    split = int(total_seqs * 0.9)
    train_idx, eval_idx = indices[:split], indices[split:]

    losses = []
    global_step = 0
    tok_start = time.time()

    for epoch in range(epochs):
        random.shuffle(train_idx)
        for i in range(0, len(train_idx) - bs, bs):
            idx = train_idx[i:i+bs]
            batch = torch.stack([tokens[j*seq_len:(j+1)*seq_len] for j in idx]).to(device)
            _, out = model(batch, labels=batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if global_step < warmup:
                lr_now = lr * (global_step + 1) / warmup
            elif global_step < decay_start:
                lr_now = lr
            else:
                p = (global_step - decay_start) / max(total_steps - decay_start, 1)
                lr_now = lr * 0.01 + 0.5 * lr * 0.99 * (1 + np.cos(np.pi * min(p, 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            losses.append(loss.item())
            global_step += 1

            if global_step <= 10 or global_step % 100 == 0:
                elapsed = time.time() - tok_start
                print(f"    step {global_step:4d}/{total_steps} | loss={loss.item():.4f} | "
                      f"ppl={np.exp(loss.item()):.0f} | lr={lr_now:.2e} | {global_step*bs*seq_len/elapsed:.0f} tok/s")

            # Evaluate
            if global_step % 200 == 0:
                model.eval()
                et, en_ = 0.0, 0
                random.shuffle(eval_idx)
                with torch.no_grad():
                    for ei in range(0, min(len(eval_idx) - bs, bs*5), bs):
                        idx_e = eval_idx[ei:ei+bs]
                        batch_e = torch.stack([tokens[j*seq_len:(j+1)*seq_len] for j in idx_e]).to(device)
                        _, eo = model(batch_e, labels=batch_e)
                        et += eo["loss"].item(); en_ += 1
                ep = np.exp(et / max(en_, 1))
                print(f"    >>> EVAL @ {global_step}: ppl={ep:.0f} <<<")
                model.train()

            if global_step >= total_steps:
                break

    # Final eval
    model.eval()
    et, en_ = 0.0, 0
    random.shuffle(eval_idx)
    with torch.no_grad():
        for ei in range(0, min(len(eval_idx) - bs, bs*8), bs):
            idx_e = eval_idx[ei:ei+bs]
            batch_e = torch.stack([tokens[j*seq_len:(j+1)*seq_len] for j in idx_e]).to(device)
            _, eo = model(batch_e, labels=batch_e)
            et += eo["loss"].item(); en_ += 1
    final_ppl = np.exp(et / max(en_, 1))
    print(f"  [{label}] Final PPL: {final_ppl:.0f} | {losses[-1]:.4f} (train)")
    model.train()
    return losses, final_ppl

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
print("="*55)
print("Phase 1 Two-Stage Training")
print("="*55)

# Stage 1: Pure template
print("\n[Stage 1] Pure template data...")
t1_texts = gen_templates(28000)
t1_tokens, t1_unique = tokenize_texts(t1_texts)
print(f"  {len(t1_texts)} texts, {len(t1_tokens):,} tokens, {t1_unique} unique types")

cfg = ModelConfig.phase1()
cfg.max_seq_len = 512
model = Transformer(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
model.train()

print(f"  Model: {cfg.total_params:,} params")
print(f"  Epochs: 8, LR: 1e-3 WSD")

s1_losses, s1_ppl = train_stage(model, opt, t1_tokens, 512, 32, 8, 1e-3, "Stage1", device)

# Stage 2: Add wiki
print(f"\n[Stage 2] Adding Wikipedia enrichment...")
t2_texts = load_wiki(1500)
t2_tokens, t2_unique = tokenize_texts(t2_texts)
print(f"  {len(t2_texts)} texts, {len(t2_tokens):,} tokens, {t2_unique} unique types")
combined = torch.cat([t1_tokens, t2_tokens])
print(f"  Combined: {len(combined):,} tokens")

s2_losses, s2_ppl = train_stage(model, opt, combined, 512, 32, 3, 5e-4, "Stage2", device)

# ── Final generation ──
print(f"\n{'='*55}")
print(f"Generation Test")
print(f"{'='*55}")

model.eval()
prompts = ["什么是模型？", "请解释神经网络", "你好，请问你今天"]
for prompt in prompts:
    ids = [1] + tokenizer.encode(prompt).ids
    prompt_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        full, new = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8, top_k=30)
    resp = tokenizer.decode(full[0].tolist(), skip_special_tokens=True)
    print(f"Q: {prompt}")
    print(f"A: {resp[:200]}")
    print()

# ── Save ──
ckpt_dir = Path("checkpoints/phase1_twostage")
ckpt_dir.mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "config": cfg.__dict__,
            "s1_ppl": s1_ppl, "s2_ppl": s2_ppl,
            "t1_unique": t1_unique, "t2_unique": t2_unique,
            "s1_losses": s1_losses, "s2_losses": s2_losses},
           ckpt_dir / "final.pt")

print(f"\n{'='*55}")
print(f"Two-Stage Complete!")
print(f"  Stage1: PPL {s1_ppl:.0f} ({t1_unique} unique tokens)")
print(f"  Stage2: PPL {s2_ppl:.0f} (+{t2_unique} wiki tokens)")
print(f"  Checkpoint: {ckpt_dir / 'final.pt'}")
print(f"✅ Done!")
