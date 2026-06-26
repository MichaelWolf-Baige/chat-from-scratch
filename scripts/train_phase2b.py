#!/usr/bin/env python
"""Phase 2b: Deep-narrow architecture. More layers, smaller d_model, same ~49M params.

Architecture comparison:
    Phase 2:  d=576, layers=11, heads=9,  d_ff=1536 → ~49M  (wide/shallow)
    Phase 2b: d=384, layers=20, heads=6,  d_ff=1024 → ~49M  (deep/narrow)

Hypothesis: More non-linear depth compensates for smaller hidden dim.
Extra layers → more abstract representations → better at juggling
template + wiki distributions without catastrophic interference.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_phase2b.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
import json
import time
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.utils.checkpoint import save_checkpoint

# ══════════════════════════════════════════════════
# DATA (same template engine + wiki loader)
# ══════════════════════════════════════════════════

CN_NOUNS = [
    "模型","算法","网络","数据","系统","代码","函数","模块","架构","接口",
    "引擎","平台","服务","应用","框架","协议","缓存","索引","线程","进程",
    "实验","理论","变量","公式","定理","假设","参数","样本","信号","能量",
    "方法","策略","方案","标准","规范","流程","机制","模式","结构","层次",
    "市场","产业","投资","金融","贸易","物流","渠道","品牌","细胞","基因",
    "组织","机构","部门","团队","项目","任务","目标","指标","周期","阶段",
]
CN_VERBS = [
    "处理","分析","计算","训练","生成","预测","优化","设计","实现","测试",
    "评估","调整","监控","部署","扩展","集成","转换","提取","检测","识别",
    "提升","降低","加速","简化","增强","改进","促进","推动","支持","保障",
]
CN_ADJS = [
    "高效","稳定","灵活","精准","可靠","先进","强大","轻量","智能","快速",
    "安全","简洁","完善","成熟","主流","创新","实用","专业","全面","显著",
    "持续","系统","深度","广泛","关键","核心","基础","必要","有效","合理",
]
EN_NOUNS = [
    "model","algorithm","network","system","data","code","function",
    "module","architecture","interface","engine","platform","framework",
    "protocol","cache","process","experiment","theory","variable","parameter",
]
EN_VERBS = [
    "process","analyze","compute","train","generate","predict",
    "optimize","design","implement","test","evaluate","deploy","integrate",
]
EN_ADJS = [
    "efficient","robust","flexible","accurate","reliable","advanced",
    "powerful","lightweight","intelligent","fast","secure","scalable",
    "significant","systematic","comprehensive","dynamic","automatic",
]

def pick(pool): return random.choice(pool)

def gen_templates(n, seed=42):
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        cn, cn2 = rng.choice(CN_NOUNS), rng.choice(CN_NOUNS)
        ca, cv = rng.choice(CN_ADJS), rng.choice(CN_VERBS)
        r = rng.random()
        if r < 0.16:
            texts.append(f"{cn}是一种{ca}的{cn2}技术，用于{cv}和{rng.choice(CN_VERBS)}，已广泛应用于{rng.choice(CN_NOUNS)}领域。研究表明{ca}的{cn}能提升{rng.choice(CN_NOUNS)}效率约{rng.choice(range(10,90))}%。")
        elif r < 0.32:
            texts.append(f"在{cn2}领域，{ca}的{cn}起到关键作用。利用{rng.choice(CN_NOUNS)}对{cn}进行{rng.choice(CN_VERBS)}可取得{ca}效果。已在{rng.choice(CN_NOUNS)}和{rng.choice(CN_NOUNS)}应用。")
        elif r < 0.46:
            texts.append(f"用户：请介绍{cn}的特点和应用。\n助手：{cn}是{ca}的{cn2}方案。核心特点：{rng.choice(CN_ADJS)}的{rng.choice(CN_NOUNS)}设计；高效{rng.choice(CN_VERBS)}能力；完善{rng.choice(CN_NOUNS)}机制。应用：{rng.choice(CN_NOUNS)}、{rng.choice(CN_NOUNS)}、{rng.choice(CN_NOUNS)}。")
        elif r < 0.60:
            texts.append(f"问题：为什么{ca}的{cn}能提升{cn2}性能？\n分析：第一，{cn}通过{rng.choice(CN_VERBS)}和{rng.choice(CN_VERBS)}处理{rng.choice(CN_NOUNS)}。第二，{cn2}瓶颈在于{rng.choice(CN_NOUNS)}效率。第三，{ca}的{cn}恰好能解决。结论：{cn}是有效的{rng.choice(CN_VERBS)}策略。")
        elif r < 0.72:
            texts.append(f"据最新报道，研究团队开发了{ca}的{cn}。该{cn}在{cn2}测试中表现{ca}，有望在{rng.choice(CN_NOUNS)}产业产生重要影响。")
        elif r < 0.82:
            en, en2 = rng.choice(EN_NOUNS), rng.choice(EN_NOUNS)
            texts.append(f"The {en} is a {rng.choice(EN_ADJS)} {en2} approach for {rng.choice(EN_VERBS)}ing {rng.choice(EN_NOUNS)}. Studies show {rng.choice(EN_ADJS)} results on {rng.choice(EN_NOUNS)} benchmarks.")
        elif r < 0.90:
            en, en2 = rng.choice(EN_NOUNS), rng.choice(EN_NOUNS)
            texts.append(f"Q: How does {en} improve {en2}?\nA: {en} uses {rng.choice(EN_ADJS)} {rng.choice(EN_NOUNS)} to {rng.choice(EN_VERBS)} {en2}, achieving {rng.choice(EN_ADJS)} accuracy. The insight is {en} can {rng.choice(EN_VERBS)} the underlying {rng.choice(EN_NOUNS)}.")
        else:
            en = rng.choice(EN_NOUNS)
            texts.append(f"class {en.capitalize()}Processor:\n    def __init__(self):\n        self.threshold = {rng.choice(range(1,10))}/10.0\n\n    def {rng.choice(EN_VERBS)}(self, data):\n        return [x for x in data if self.score(x) > self.threshold]\n\n    def score(self, item):\n        return sum(ord(c) % 100 for c in str(item)) / 100.0")
    return texts

def load_wiki(n):
    texts = []
    with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line); text = obj.get("text", "")
            if 80 <= len(text) <= 300: texts.append(text)
            if len(texts) >= n: break
    return texts

def tokenize(texts, tok_path):
    from tokenizers import Tokenizer as HFT
    tok = HFT.from_file(tok_path)
    all_ids = []
    for text in texts:
        ids = tok.encode(text).ids
        all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    return torch.tensor(all_ids, dtype=torch.long)


# ══════════════════════════════════════════════════
# WSD SCHEDULER
# ══════════════════════════════════════════════════

class WSDScheduler:
    def __init__(self, opt, warmup, decay_start, total, max_lr):
        self.opt = opt; self.w = warmup; self.ds = decay_start; self.T = total
        self.max_lr = max_lr
    def step(self, step):
        if step < self.w:
            lr = self.max_lr * (step + 1) / self.w
        elif step < self.ds:
            lr = self.max_lr
        else:
            p = min((step - self.ds) / max(self.T - self.ds, 1), 1.0)
            lr = self.max_lr * 0.01 + 0.5 * self.max_lr * (1.0 + np.cos(np.pi * p))
        for pg in self.opt.param_groups: pg["lr"] = lr
        return lr


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local_r = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_r); device = torch.device(f"cuda:{local_r}")
    seed = 42 + rank
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(seed)

    if rank == 0:
        print("=" * 55)
        print("Phase 2b: Deep-Narrow Architecture")
        print("=" * 55)

    # ── Data ──
    t1_texts = gen_templates(40000, seed=42)
    t1_tokens = tokenize(t1_texts, "tokenizers/phase1_8k_real/tokenizer.json")
    if rank == 0:
        print(f"[Data] Templates: {len(t1_texts):,} samples, {len(t1_tokens):,} tokens")

    # ── Model: deep + narrow ──
    # Parameter check:
    #   Embedding: 8192*384 = 3.15M
    #   Per layer: 4*384² (attn) + 3*384*1024 (FFN) + 2*384 (norm) = 590K+1.18M+768 = 1.77M
    #   20 layers: 20*1.77M = 35.4M
    #   Total: 3.15M + 35.4M + 384 = ~38.6M  -- a bit low
    # Adjust: d_ff=1280 → per layer attn 590K + FFN 3*384*1280=1.47M = 2.06M
    #   20 layers = 41.3M → total ~44.4M
    # Try: d_ff=1408 → per layer FFN 3*384*1408=1.62M, attn 590K = 2.21M
    #   20*2.21M=44.2M, total ~47.4M ✓
    cfg = ModelConfig(
        vocab_size=8192, d_model=384, n_layers=20, n_heads=6, n_kv_heads=6,
        d_ff=1408, max_seq_len=512, rope_theta=10000.0,
        dropout=0.0, use_flash_attention=True, tie_word_embeddings=True,
        rms_norm_eps=1e-6, initializer_range=0.02,
    )
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_r], find_unused_parameters=False,
                gradient_as_bucket_view=True)
    model.train()

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"[Model] {n:,} params | d={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} d_ff={cfg.d_ff}")
        print(f"         d_head={cfg.d_head} | depth/width ratio = {cfg.n_layers}/{cfg.d_model} = {cfg.n_layers/cfg.d_model:.3f}")
        print(f"         (Phase 2 was d=576 layers=11 → ratio = 0.019)")

    # ── Training setup ──
    seq_len = 512; bs = 24; epochs_s1 = 10; max_lr = 8e-4
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)

    def train_stage(tokens, epochs, label, lr_scale=1.0):
        total_seqs = len(tokens) // seq_len
        steps_per_epoch = total_seqs // (bs * world)
        total_steps = steps_per_epoch * epochs
        warmup = total_steps // 8; decay_start = int(total_steps * 0.85)
        sched = WSDScheduler(opt, warmup, decay_start, total_steps, max_lr * lr_scale)

        split = int(len(tokens) * 0.95)
        train_t, eval_t = tokens[:split], tokens[split:]
        train_ds = PretrainDataset(train_t, seq_len=seq_len)
        eval_ds = PretrainDataset(eval_t, seq_len=seq_len)
        train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        eval_s = DistributedSampler(eval_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
        train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                               num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        eval_l = torch.utils.data.DataLoader(eval_ds, batch_size=bs, sampler=eval_s,
                                              num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

        global_step = 0; losses = []; tok_start = time.time()
        for epoch in range(epochs):
            train_s.set_epoch(epoch)
            for batch in train_l:
                if global_step >= total_steps: break
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                _, out = model(input_ids, labels=labels)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(global_step); opt.zero_grad()
                loss_val = loss.item(); global_step += 1

                if rank == 0:
                    losses.append(loss_val)
                    if global_step <= 20 or global_step % 100 == 0:
                        elapsed = time.time() - tok_start
                        lr = opt.param_groups[0]["lr"]
                        tps = global_step * bs * world * seq_len / max(elapsed, 0.01)
                        print(f"  [{label}] step {global_step:4d}/{total_steps} | loss={loss_val:.4f} | "
                              f"ppl={np.exp(loss_val):.0f} | lr={lr:.2e} | {tps/1000:.0f}K tok/s")

                    if global_step % 300 == 0:
                        model.eval(); et, en_ = 0.0, 0
                        with torch.no_grad():
                            for ei, eb in enumerate(eval_l):
                                if ei >= 8: break
                                ein, ela = eb["input_ids"].to(device), eb["labels"].to(device)
                                _, eo = model(ein, labels=ela); et += eo["loss"].item(); en_ += 1
                        print(f"  >>> [{label}] EVAL @ {global_step}: ppl={np.exp(et/max(en_,1)):.0f} <<<")
                        model.train()
        # Final
        model.eval(); et, en_ = 0.0, 0
        with torch.no_grad():
            for ei, eb in enumerate(eval_l):
                if ei >= 15: break
                ein, ela = eb["input_ids"].to(device), eb["labels"].to(device)
                _, eo = model(ein, labels=ela); et += eo["loss"].item(); en_ += 1
        final_ppl = np.exp(et / max(en_, 1))
        if rank == 0: print(f"  [{label}] Final PPL: {final_ppl:.0f} | {(time.time()-tok_start)/60:.1f}min")
        model.train(); return losses, final_ppl

    # ══════ Stage 1: Template ══════
    if rank == 0: print(f"\n[Stage 1] Template foundation ({epochs_s1} epochs)")
    s1_losses, s1_ppl = train_stage(t1_tokens, epochs_s1, "S1")

    # ══════ Stage 2: Wiki ══════
    if rank == 0: print(f"\n[Stage 2] Wikipedia enrichment")
    wiki_texts = load_wiki(4000)
    t2_tokens = tokenize(wiki_texts, "tokenizers/phase1_8k_real/tokenizer.json")
    combined = torch.cat([t1_tokens, t2_tokens])
    if rank == 0: print(f"  Wiki: {len(wiki_texts):,} samples | Combined: {len(combined):,} tokens")
    s2_losses, s2_ppl = train_stage(combined, 5, "S2", lr_scale=0.5)

    # ══════ Generation ══════
    if rank == 0:
        from tokenizers import Tokenizer as HFT
        tok = HFT.from_file("tokenizers/phase1_8k_real/tokenizer.json")
        print(f"\n{'='*55}\nGeneration Demo\n{'='*55}")
        model.eval()
        for prompt in ["什么是模型？","请解释一下神经网络","你好，请问"]:
            ids = [1] + tok.encode(prompt).ids
            pid = torch.tensor([ids], device=device)
            with torch.no_grad():
                full, new = model.module.generate(pid, max_new_tokens=50, temperature=0.8, top_k=35, top_p=0.9)
            print(f"  Q: {prompt}")
            print(f"  A: {tok.decode(full[0].tolist(), skip_special_tokens=True)[:200]}\n")

        # Save
        ckpt_dir = Path("checkpoints/phase2b")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_dir / "final.pt", model.module, opt, None, step=0, epoch=0,
                        config={"arch":"deep-narrow","d_model":cfg.d_model,"n_layers":cfg.n_layers,
                                "s1_ppl":float(s1_ppl),"s2_ppl":float(s2_ppl)})
        print(f"Phase 2b Complete! S1 PPL={s1_ppl:.0f} S2 PPL={s2_ppl:.0f}")
        print(f"Checkpoint: {ckpt_dir / 'final.pt'}")
        print("✅ Done!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
