#!/usr/bin/env python
"""Phase 2: 49M model, 4-GPU DDP, two-stage training.

Lessons baked in from Phase 1:
  - Token rarity is THE bottleneck → template foundation with controlled vocab
  - Cosine decay kills small-model training → WSD schedule (80% stable)
  - fp16/bf16 gradients fine for 14M → same for 49M
  - DDP 4-card scales near-linearly (3.8x)
  - DataLoader workers=2 with persistent_workers=True for DDP
  - Two-stage: template first (learn patterns) → wiki enrichment (add breadth)

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_phase2.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import torch.nn.functional as F
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

# ═══════════════════════════════════════════════════════════════════
# DATA ENGINE — Phase 2 scale (richer templates + more wiki)
# ═══════════════════════════════════════════════════════════════════

CN_NOUNS = [
    "模型","算法","网络","数据","系统","代码","函数","模块","架构","接口",
    "引擎","平台","服务","应用","框架","协议","缓存","索引","线程","进程",
    "实验","理论","变量","公式","定理","假设","参数","样本","信号","能量",
    "学校","医院","城市","交通","能源","食品","水源","气候","人口","经济",
    "方法","策略","方案","标准","规范","流程","机制","模式","结构","层次",
    "市场","产业","投资","金融","贸易","物流","供应链","渠道","品牌","消费者",
    "细胞","基因","蛋白质","分子","原子","电子","磁场","光谱","温度","压力",
    "组织","机构","部门","团队","项目","任务","目标","指标","周期","阶段",
]

CN_VERBS = [
    "处理","分析","计算","训练","生成","预测","优化","设计","实现","测试",
    "评估","调整","监控","部署","扩展","集成","转换","提取","检测","识别",
    "提升","降低","加速","简化","增强","改进","促进","推动","支持","保障",
    "建立","构建","创建","开发","制定","执行","管理","维护","更新","升级",
]

CN_ADJS = [
    "高效","稳定","灵活","精准","可靠","先进","强大","轻量","智能","快速",
    "安全","简洁","完善","成熟","主流","创新","实用","通用","专业","全面",
    "显著","持续","系统","深度","广泛","严格","精细","自动","动态","均衡",
    "关键","核心","基础","重要","必要","有效","合理","科学","规范","标准",
]

EN_NOUNS = [
    "model","algorithm","network","system","data","code","function",
    "module","architecture","interface","engine","platform","service",
    "framework","protocol","cache","index","thread","process",
    "experiment","theory","variable","formula","theorem","parameter",
    "cell","gene","protein","molecule","atom","electron","magnetic",
]

EN_VERBS = [
    "process","analyze","compute","train","generate","predict",
    "optimize","design","implement","test","evaluate","deploy",
    "accelerate","enhance","integrate","transform","extract","detect",
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
        ca = rng.choice(CN_ADJS); cv = rng.choice(CN_VERBS)
        r = rng.random()
        if r < 0.16:
            s = f"{cn}是一种{ca}的{cn2}技术，主要用于{cv}和{rng.choice(CN_VERBS)}。该技术在{rng.choice(range(2018,2027))}年首次提出，经过多年发展已广泛应用于{rng.choice(CN_NOUNS)}领域。研究表明{rng.choice(CN_ADJS)}的{cn}能够将{rng.choice(CN_NOUNS)}效率提升约{rng.choice(range(10,90))}%。"
        elif r < 0.32:
            s = f"在{cn2}领域，{ca}的{cn}起到关键作用。研究显示，利用{rng.choice(CN_NOUNS)}对{cn}进行{rng.choice(CN_VERBS)}，可以取得{ca}的效果。目前该方法已在{rng.choice(CN_NOUNS)}和{rng.choice(CN_NOUNS)}等领域得到应用。"
        elif r < 0.46:
            s = f"用户：请介绍{cn}的主要特点和应用。\n助手：{cn}是一种{ca}的{cn2}方案。核心特点包括三点：第一，{rng.choice(CN_ADJS)}的{rng.choice(CN_NOUNS)}设计；第二，高效的{rng.choice(CN_VERBS)}能力；第三，完善的{rng.choice(CN_NOUNS)}机制。应用场景涵盖{rng.choice(CN_NOUNS)}、{rng.choice(CN_NOUNS)}和{rng.choice(CN_NOUNS)}。"
        elif r < 0.60:
            s = f"问题：为什么{ca}的{cn}能够提升{cn2}的性能？\n分析：第一步，了解{cn}原理——它通过{rng.choice(CN_VERBS)}和{rng.choice(CN_VERBS)}来处理{rng.choice(CN_NOUNS)}。第二步，分析{cn2}瓶颈——主要问题在于{rng.choice(CN_NOUNS)}的{rng.choice(CN_VERBS)}效率低。第三步，{ca}的{cn}恰好能解决这一问题。结论：{cn}是一种有效的{rng.choice(CN_VERBS)}策略。"
        elif r < 0.72:
            s = f"据最新报道，研究团队成功开发了一种{ca}的{cn}。该{cn}在{cn2}测试中表现{ca}，有望在{rng.choice(CN_NOUNS)}产业产生重要影响。项目负责人表示，{cn}将改变{rng.choice(CN_NOUNS)}的未来发展方向。"
        elif r < 0.82:
            en, en2 = rng.choice(EN_NOUNS), rng.choice(EN_NOUNS)
            s = f"The {en} is a {rng.choice(EN_ADJS)} {en2} approach designed to {rng.choice(EN_VERBS)} and {rng.choice(EN_VERBS)} {rng.choice(EN_NOUNS)}. Studies demonstrate that combining {en} with {rng.choice(EN_NOUNS)} yields {rng.choice(EN_ADJS)} results on {rng.choice(EN_NOUNS)} benchmarks."
        elif r < 0.90:
            en, en2 = rng.choice(EN_NOUNS), rng.choice(EN_NOUNS)
            s = f"Q: How does {en} improve {en2}?\nA: {en} leverages {rng.choice(EN_ADJS)} {rng.choice(EN_NOUNS)} to {rng.choice(EN_VERBS)} {en2}, achieving {rng.choice(EN_ADJS)} accuracy. The core innovation is that {en} can {rng.choice(EN_VERBS)} the underlying {rng.choice(EN_NOUNS)} more effectively than prior approaches."
        else:
            en = rng.choice(EN_NOUNS)
            s = f"class {en.capitalize()}Processor:\n    def __init__(self, config=None):\n        self.config = config or {{'lr': 0.001}}\n\n    def {rng.choice(EN_VERBS)}(self, data):\n        results = []\n        for item in data:\n            score = self.compute_score(item)\n            if score > self.config.get('threshold', 0.5):\n                results.append(item)\n        return results\n\n    def compute_score(self, item):\n        return sum(hash(c) % 100 for c in str(item)) / 100.0"
        texts.append(s)
    return texts

def load_wiki(n):
    texts = []
    with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            if 80 <= len(text) <= 300:
                texts.append(text)
            if len(texts) >= n:
                break
    return texts


def tokenize(texts, tokenizer_path):
    from tokenizers import Tokenizer as HFT
    tok = HFT.from_file(tokenizer_path)
    all_ids = []
    for text in texts:
        ids = tok.encode(text).ids
        all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    return torch.tensor(all_ids, dtype=torch.long), len(torch.unique(torch.tensor(all_ids)))


# ═══════════════════════════════════════════════════════════════════
# WSD SCHEDULER
# ═══════════════════════════════════════════════════════════════════

class WSDScheduler:
    """Warmup-Stable-Decay LR schedule."""
    def __init__(self, opt, warmup, stable_start, decay_start, total, max_lr):
        self.opt = opt; self.w = warmup
        self.ss = stable_start; self.ds = decay_start; self.T = total
        self.max_lr = max_lr

    def step(self, step):
        if step < self.w:
            lr = self.max_lr * (step + 1) / self.w
        elif step < self.ds:
            lr = self.max_lr
        else:
            p = min((step - self.ds) / max(self.T - self.ds, 1), 1.0)
            lr = self.max_lr * 0.01 + 0.5 * self.max_lr * (1.0 + np.cos(np.pi * p))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = 42 + rank
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

    # ── Generate data (all ranks generate same data for reproducibility) ──
    if rank == 0:
        print("=" * 55)
        print("Phase 2: 49M model, 4-GPU DDP, Two-Stage")
        print("=" * 55)
        print("[Stage 1] Generating template data...")

    t1_texts = gen_templates(40000, seed=42)
    t1_tokens, _ = tokenize(t1_texts, "tokenizers/phase1_8k_real/tokenizer.json")

    if rank == 0:
        print(f"  Templates: {len(t1_texts):,} samples, {len(t1_tokens):,} tokens")

    # ── Model ──
    # Phase 2 architecture with 8K vocab (keeps existing tokenizer)
    cfg = ModelConfig(
        vocab_size=8192, d_model=576, n_layers=11, n_heads=9, n_kv_heads=9,
        d_ff=1536, max_seq_len=512, rope_theta=10000.0,
        dropout=0.0, use_flash_attention=True,
        rms_norm_eps=1e-6, tie_word_embeddings=True, initializer_range=0.02,
    )
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_rank],
                find_unused_parameters=False, gradient_as_bucket_view=True)
    model.train()

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n:,} params (target ~49M)")
        print(f"  Arch: d={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, d_ff={cfg.d_ff}")
        print(f"  Global batch: {32*world_size}x512")

    # ── Training setup ──
    seq_len = 512
    bs = 26  # per-GPU (global = 26*4 = 104)
    epochs_s1 = 10
    max_lr = 8e-4
    grad_accum = 1

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95),
                             weight_decay=0.1, fused=False)

    def train_stage(tokens, epochs, stage_label, lr_scale=1.0):
        nonlocal opt
        # Reset optimizer for new stage
        for pg in opt.param_groups: pg["lr"] = max_lr * lr_scale

        total_seqs = len(tokens) // seq_len
        steps_per_epoch = total_seqs // (bs * world_size)
        total_steps = steps_per_epoch * epochs

        warmup = total_steps // 8
        decay_start = int(total_steps * 0.85)
        sched = WSDScheduler(opt, warmup, warmup, decay_start, total_steps, max_lr * lr_scale)

        # Datasets with distributed sampler
        train_t = tokens[:int(len(tokens)*0.95)]
        eval_t = tokens[int(len(tokens)*0.95):]

        train_ds = PretrainDataset(train_t, seq_len=seq_len)
        eval_ds = PretrainDataset(eval_t, seq_len=seq_len)

        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        eval_sampler = DistributedSampler(
            eval_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=bs, sampler=train_sampler,
            num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        eval_loader = torch.utils.data.DataLoader(
            eval_ds, batch_size=bs, sampler=eval_sampler,
            num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

        global_step = 0
        losses = []
        tok_start = time.time()

        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if global_step >= total_steps:
                    break

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                _, out = model(input_ids, labels=labels)
                loss = out["loss"] / grad_accum
                loss.backward()

                if (global_step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    sched.step(global_step)

                loss_val = out["loss"].item()
                global_step += 1

                if rank == 0:
                    losses.append(loss_val)
                    if global_step <= 20 or global_step % 100 == 0:
                        elapsed = time.time() - tok_start
                        lr = opt.param_groups[0]["lr"]
                        tokens_done = global_step * bs * world_size * seq_len
                        print(f"  [{stage_label}] step {global_step:4d}/{total_steps} | "
                              f"loss={loss_val:.4f} ppl={np.exp(loss_val):.0f} | "
                              f"lr={lr:.2e} | {tokens_done/elapsed:.0f} tok/s")

                    if global_step % 300 == 0:
                        model.eval()
                        et, en_ = 0.0, 0
                        with torch.no_grad():
                            for ei, eb in enumerate(eval_loader):
                                if ei >= 8: break
                                ein = eb["input_ids"].to(device)
                                ela = eb["labels"].to(device)
                                _, eo = model(ein, labels=ela)
                                et += eo["loss"].item(); en_ += 1
                        ep = np.exp(et / max(en_, 1))
                        print(f"  >>> [{stage_label}] EVAL @ {global_step}: ppl={ep:.0f} <<<")
                        model.train()

            if global_step >= total_steps:
                break

        # Final eval
        model.eval()
        et, en_ = 0.0, 0
        with torch.no_grad():
            for ei, eb in enumerate(eval_loader):
                if ei >= 15: break
                ein = eb["input_ids"].to(device)
                ela = eb["labels"].to(device)
                _, eo = model(ein, labels=ela)
                et += eo["loss"].item(); en_ += 1
        final_ppl = np.exp(et / max(en_, 1))

        if rank == 0:
            print(f"  [{stage_label}] Final PPL: {final_ppl:.0f} | "
                  f"Time: {(time.time()-tok_start)/60:.1f}min")
        model.train()
        return losses, final_ppl

    # ══════════════ STAGE 1: Template foundation ══════════════
    if rank == 0:
        print(f"\n[Stage 1] Template foundation ({epochs_s1} epochs)")
    s1_losses, s1_ppl = train_stage(t1_tokens, epochs_s1, "S1")

    # ══════════════ STAGE 2: Wiki enrichment ══════════════
    if rank == 0:
        print(f"\n[Stage 2] Wikipedia enrichment")
        print("Loading Wikipedia data...")

    wiki_texts = load_wiki(4000)
    t2_tokens, _ = tokenize(wiki_texts, "tokenizers/phase1_8k_real/tokenizer.json")

    if rank == 0:
        print(f"  Wiki: {len(wiki_texts):,} samples, {len(t2_tokens):,} tokens")

    combined = torch.cat([t1_tokens, t2_tokens])

    if rank == 0:
        print(f"  Combined: {len(combined):,} tokens total")

    s2_losses, s2_ppl = train_stage(combined, 5, "S2", lr_scale=0.5)

    # ══════════════ GENERATION ══════════════
    if rank == 0:
        print(f"\n{'='*55}")
        print("Generation Demo")
        print(f"{'='*55}")

        model.eval()
        from tokenizers import Tokenizer as HFT
        tok = HFT.from_file("tokenizers/phase1_8k_real/tokenizer.json")

        prompts = ["什么是模型？", "请解释一下神经网络", "你好，请问"]
        for prompt in prompts:
            ids = [1] + tok.encode(prompt).ids
            prompt_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                full, new = model.module.generate(
                    prompt_ids, max_new_tokens=50, temperature=0.8, top_k=35, top_p=0.9)
            resp = tok.decode(full[0].tolist(), skip_special_tokens=True)
            print(f"  Q: {prompt}")
            print(f"  A: {resp[:200]}")
            print()

        # ── Save ──
        ckpt_dir = Path("checkpoints/phase2")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_dir / "final.pt", model.module, opt, None,
                        step=0, epoch=0, config={
                            "model_params": cfg.total_params,
                            "stage1_ppl": float(s1_ppl),
                            "stage2_ppl": float(s2_ppl),
                            "stage1_samples": len(t1_texts),
                            "stage2_samples": len(wiki_texts),
                        })

        print(f"\n{'='*55}")
        print(f"Phase 2 Complete!")
        print(f"  Stage1 PPL: {s1_ppl:.0f}")
        print(f"  Stage2 PPL: {s2_ppl:.0f}")
        print(f"  Checkpoint: {ckpt_dir / 'final.pt'}")
        print(f"✅ Done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
