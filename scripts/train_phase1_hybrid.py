#!/usr/bin/env python
"""Phase 1 Hybrid: Template core + Real Wikipedia enrichment.

Lessons from experiments:
- Pure Wiki (8192 types):     PPL 2200 — tokens too rare
- Pure Template (831 types):   PPL 1    — data too narrow (memorization)
- Hybrid (4000-6000 types):    PPL ???  — the right balance

Strategy:
1. Generate 15000 template samples (consistent core vocabulary)
2. Add 3000 curated Wikipedia paragraphs (real distribution)
3. 8 epochs with WSD schedule
Expected token variety: ~3500-5000 unique across 8192 vocab
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import time
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.tokenizer_utils import load_tokenizer

device = torch.device("cuda:0")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = load_tokenizer("tokenizers/phase1_8k_real/tokenizer.json")

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Template data (from exp_f — high repetition core)
# ═══════════════════════════════════════════════════════════════════
print("=" * 55)
print("Phase 1 HYBRID: Template core + Wikipedia enrichment")
print("=" * 55)

# -- same template engine from train_phase1_final.py but larger entity pools
random.seed(42)
cn_nouns = [
    "模型","算法","网络","数据","系统","代码","函数","模块","架构","接口",
    "引擎","平台","服务","应用","框架","协议","缓存","索引","线程","进程",
    "实验","理论","变量","公式","定理","假设","参数","样本","信号","能量",
    "学校","医院","城市","交通","能源","食品","水源","气候","人口","经济",
    "方法","策略","方案","标准","规范","流程","机制","模式","结构","层次",
    "市场","产业","投资","金融","贸易","物流","供应链","渠道","品牌","消费者",
    "细胞","基因","蛋白质","分子","原子","电子","磁场","光谱","温度","压力",
]
cn_verbs = [
    "处理","分析","计算","训练","生成","预测","优化","设计","实现","测试",
    "评估","调整","监控","部署","扩展","集成","转换","提取","检测","识别",
    "提升","降低","加速","简化","增强","改进","促进","推动","支持","保障",
]
cn_adjs = [
    "高效","稳定","灵活","精准","可靠","先进","强大","轻量","智能","快速",
    "安全","简洁","完善","成熟","主流","创新","实用","通用","专业","全面",
    "显著","持续","系统","深度","广泛","严格","精细","自动","动态","均衡",
]
en_nouns = [
    "model","algorithm","network","system","data","code","function",
    "module","architecture","interface","engine","platform","service",
    "framework","protocol","cache","index","thread","process",
    "experiment","theory","variable","formula","theorem","parameter",
    "cell","gene","protein","molecule","atom","electron","magnetic","spectrum",
]
en_verbs = [
    "process","analyze","compute","train","generate","predict",
    "optimize","design","implement","test","evaluate","deploy",
    "accelerate","enhance","integrate","transform","extract","detect",
]
en_adjs = [
    "efficient","robust","flexible","accurate","reliable","advanced",
    "powerful","lightweight","intelligent","fast","secure","scalable",
    "significant","systematic","comprehensive","dynamic","automatic","deep",
]
years = list(range(2018, 2027))
nums = list(range(1, 100))
countries = ["中国","美国","日本","德国","法国","英国","韩国","印度","巴西","加拿大"]

def pick(lst):
    return random.choice(lst)

def gen_templates(n):
    texts = []
    for i in range(n):
        s, s2 = pick(cn_nouns), pick(cn_nouns)
        a = pick(cn_adjs)
        v = pick(cn_verbs)
        r = random.random()
        if r < 0.18:
            texts.append(f"{s}是一种{a}的{s2}技术，主要用于{v}和{pick(cn_verbs)}。该技术在{pick(years)}年首次提出，已广泛应用于{pick(cn_nouns)}领域。")
        elif r < 0.36:
            texts.append(f"在{s2}领域，{a}的{s}起到关键作用。研究显示利用{pick(cn_nouns)}对{s}进行{pick(cn_verbs)}可取得{a}的效果。")
        elif r < 0.50:
            texts.append(f"据{pick(countries)}最新报道，研究团队开发了{a}的{s}，该{s}在{s2}测试中表现{a}。")
        elif r < 0.62:
            texts.append(f"用户：什么是{s}？请详细说明。\n助手：{s}是一种{a}的{s2}方案，核心优势包括{pick(cn_adjs)}的{pick(cn_nouns)}、{pick(cn_adjs)}的{pick(cn_verbs)}能力以及完善的{pick(cn_nouns)}。它广泛应用于{pick(cn_nouns)}和{pick(cn_nouns)}。")
        elif r < 0.74:
            s_en, s2_en = pick(en_nouns), pick(en_nouns)
            texts.append(f"The {s_en} is a {pick(en_adjs)} {s2_en} approach designed to {pick(en_verbs)} and {pick(en_verbs)} {pick(en_nouns)}. First proposed in {pick(years)}, it has influenced {pick(en_nouns)} research significantly.")
        elif r < 0.85:
            s_en, s2_en = pick(en_nouns), pick(en_nouns)
            texts.append(f"Q: How does {s_en} improve {s2_en}?\nA: {s_en} uses {pick(en_adjs)} {pick(en_nouns)} to {pick(en_verbs)} {s2_en}, resulting in {pick(en_adjs)} performance on {pick(en_nouns)} tasks.")
        elif r < 0.93:
            texts.append(f"问题：为什么{a}的{s}能提升{s2}性能？\n分析：第一步，理解{s}的原理——通过{pick(cn_verbs)}和{pick(cn_verbs)}处理{pick(cn_nouns)}。第二步，分析{s2}瓶颈——问题在于{pick(cn_nouns)}效率低。第三步，{a}的{s}恰好能{pick(cn_verbs)}这一瓶颈。结论：{a}的{s}是一种有效的{pick(cn_verbs)}策略。")
        else:
            s_en = pick(en_nouns)
            texts.append(f"def {pick(en_verbs)}_{s_en}(data, threshold={pick(nums)}):\n    results = [x for x in data if {s_en}_score(x) > threshold]\n    return sorted(results, reverse=True)\n\nclass {s_en.capitalize()}Processor:\n    def __init__(self):\n        self.config = {{'mode': '{pick(en_verbs)}'}}\n    def process(self, inputs):\n        return [{pick(en_verbs)}(x) for x in inputs]")
    return texts

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Wikipedia enrichment — add real distribution
# ═══════════════════════════════════════════════════════════════════
print("Loading Wikipedia samples...")
wiki_texts = []
with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get("text", "")
        if 80 <= len(text) <= 400:
            wiki_texts.append(text)
        if len(wiki_texts) >= 3000:
            break
print(f"  Loaded {len(wiki_texts)} Wikipedia paragraphs")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Combine + Tokenize
# ═══════════════════════════════════════════════════════════════════
print("Generating template data...")
template_texts = gen_templates(15000)
all_texts = template_texts + wiki_texts
random.shuffle(all_texts)

print(f"Total: {len(all_texts):,} texts ({len(template_texts):,} template + {len(wiki_texts):,} wiki)")

all_ids = []
for text in all_texts:
    ids = tokenizer.encode(text).ids
    all_ids.append(1)
    all_ids.extend(ids)
    all_ids.append(2)

tokens = torch.tensor(all_ids, dtype=torch.long)
unique_tok = len(torch.unique(tokens))
print(f"Tokens: {len(tokens):,} total, {unique_tok} unique ({unique_tok/8192:.1%} of vocab)")
print(f"Avg tokens/text: {len(tokens)/len(all_texts):.0f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Model + Training
# ═══════════════════════════════════════════════════════════════════
cfg = ModelConfig.phase1()
cfg.max_seq_len = 512
model = Transformer(cfg).to(device)
n_params = cfg.total_params

seq_len = 512
total_seqs = len(tokens) // seq_len
bs = 32
epochs = 10
steps_per_epoch = total_seqs // bs
total_steps = steps_per_epoch * epochs
lr = 1e-3
warmup = total_steps // 10

opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
decay_start = int(total_steps * 0.85)

split = int(total_seqs * 0.9)
train_indices = list(range(split))
eval_indices = list(range(split, total_seqs))

print(f"\n  Model: {n_params:,} params | seq={seq_len} bs={bs}")
print(f"  Epochs: {epochs} | Steps: ~{total_steps} | LR: {lr} WSD")
print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")

model.train()
losses = []
eval_losses = []
global_step = 0
tok_start = time.time()

for epoch in range(epochs):
    random.shuffle(train_indices)
    for i in range(0, len(train_indices) - bs, bs):
        idx = train_indices[i:i + bs]
        batch = torch.stack([tokens[j * seq_len:(j + 1) * seq_len] for j in idx]).to(device)
        _, outputs = model(batch, labels=batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if global_step < warmup:
            lr_now = lr * (global_step + 1) / warmup
        elif global_step < decay_start:
            lr_now = lr
        else:
            progress = (global_step - decay_start) / max(total_steps - decay_start, 1)
            lr_now = lr * 0.01 + 0.5 * lr * 0.99 * (1 + np.cos(np.pi * min(progress, 1.0)))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        losses.append(loss.item())
        global_step += 1

        if global_step <= 20 or global_step % 100 == 0:
            elapsed = time.time() - tok_start
            print(f"  step {global_step:5d} | e{epoch+1} | loss={loss.item():.4f} | "
                  f"ppl={np.exp(loss.item()):5.0f} | lr={lr_now:.2e} | {global_step*bs*seq_len/elapsed:.0f} tok/s")

        if global_step % 500 == 0:
            model.eval()
            et, en_ = 0.0, 0
            with torch.no_grad():
                random.shuffle(eval_indices)
                for ei in range(0, min(len(eval_indices) - bs, bs * 5), bs):
                    idx_e = eval_indices[ei:ei + bs]
                    batch_e = torch.stack([tokens[j * seq_len:(j + 1) * seq_len] for j in idx_e]).to(device)
                    _, eo = model(batch_e, labels=batch_e)
                    et += eo["loss"].item()
                    en_ += 1
            ep = np.exp(et / max(en_, 1))
            eval_losses.append((global_step, et / max(en_, 1), ep))
            print(f"  >>> EVAL @ {global_step}: loss={et/max(en_,1):.4f} ppl={ep:.0f} <<<")
            model.train()

# ── Final ──
model.eval()
et, en_ = 0.0, 0
with torch.no_grad():
    random.shuffle(eval_indices)
    for ei in range(0, min(len(eval_indices) - bs, bs * 10), bs):
        idx_e = eval_indices[ei:ei + bs]
        batch_e = torch.stack([tokens[j * seq_len:(j + 1) * seq_len] for j in idx_e]).to(device)
        _, eo = model(batch_e, labels=batch_e)
        et += eo["loss"].item()
        en_ += 1
final_ppl = np.exp(et / max(en_, 1))
elapsed = time.time() - tok_start

print(f"\n{'='*55}")
print(f"Phase 1 Hybrid Complete!")
print(f"  Data: {len(all_texts):,} texts ({unique_tok} unique tokens)")
print(f"  Steps: {global_step} | Time: {elapsed/60:.1f}min")
print(f"  Train: loss {losses[0]:.2f}->{losses[-1]:.4f} | ppl {np.exp(losses[0]):.0f}->{np.exp(losses[-1]):.0f}")
print(f"  Eval PPL: {final_ppl:.0f}")
for gs, el, ep in eval_losses:
    print(f"    @{gs}: ppl={ep:.0f}")

# ── Generate ──
print(f"\n{'='*55}")
print("Generation test")
print(f"{'='*55}")
prompts = ["什么是模型？", "解释神经网络", "你好，请问"]
for prompt in prompts:
    prompt_ids = torch.tensor([[1] + tokenizer.encode(prompt).ids], device=device)
    with torch.no_grad():
        full, new = model.generate(prompt_ids, max_new_tokens=40, temperature=0.8)
    resp = tokenizer.decode(full[0].tolist(), skip_special_tokens=True)
    print(f"  Q: {prompt}")
    print(f"  A: {resp[:150]}")
    print()

# ── Save ──
ckpt_dir = Path("checkpoints/phase1_hybrid")
ckpt_dir.mkdir(parents=True, exist_ok=True)
from src.utils.checkpoint import save_checkpoint
save_checkpoint(ckpt_dir / "final.pt", model, opt, None, step=global_step, epoch=0, config={})

summary = {
    "strategy": "hybrid",
    "template_samples": len(template_texts),
    "wiki_samples": len(wiki_texts),
    "total_texts": len(all_texts),
    "total_tokens": len(tokens),
    "unique_tokens": unique_tok,
    "epochs": epochs,
    "total_steps": global_step,
    "train_loss_start_end": [losses[0], losses[-1]],
    "eval_ppl": float(final_ppl),
    "time_sec": elapsed,
}
with open(ckpt_dir / "run_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Checkpoint: {ckpt_dir / 'final.pt'}")
