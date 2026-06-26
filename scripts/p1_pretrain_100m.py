#!/usr/bin/env python
"""P1: 100M Deep-Narrow Pretraining — Template + Wiki, 4-GPU DDP.

Arch: d=512, L=24, GQA 2:1, QK-Norm, ~99M params.
Data: 60% template + 30% wiki + 10% natural text.
Target: 500M-1B tokens, WSD schedule.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/p1_pretrain_100m.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np, random, time, json
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.utils.checkpoint import save_checkpoint

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# DATA ENGINE: Template + Wiki + Natural Text
# ═══════════════════════════════════════════════════════════════

CN_NOUNS = [
    "模型","算法","网络","数据","系统","代码","函数","模块","架构","接口",
    "引擎","平台","服务","应用","框架","协议","缓存","索引","线程","进程",
    "实验","理论","变量","公式","定理","假设","参数","样本","信号","能量",
    "方法","策略","方案","标准","规范","流程","机制","模式","结构","层次",
    "市场","产业","投资","金融","贸易","物流","渠道","品牌","细胞","基因",
    "组织","机构","部门","团队","项目","任务","目标","指标","周期","阶段",
    "分子","原子","电子","磁场","光谱","温度","压力","气候","能源","城市",
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
YRS = list(range(2018,2027)); NMS = list(range(1,100))

def tp(pool): return random.choice(pool)

def gen_templates(n, seed=42):
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        cn,cn2 = rng.choice(CN_NOUNS),rng.choice(CN_NOUNS)
        ca,cv = rng.choice(CN_ADJS),rng.choice(CN_VERBS)
        r = rng.random()
        if r < 0.14:
            texts.append(f"{cn}是一种{ca}的{cn2}技术，用于{cv}和{tp(CN_VERBS)}，已广泛应用于{tp(CN_NOUNS)}领域。研究表明{ca}的{cn}能提升{tp(CN_NOUNS)}效率约{tp(NMS)}%。")
        elif r < 0.28:
            texts.append(f"在{cn2}领域，{ca}的{cn}起到关键作用。利用{tp(CN_NOUNS)}对{cn}进行{tp(CN_VERBS)}可取得{ca}效果。已在{tp(CN_NOUNS)}和{tp(CN_NOUNS)}应用。该方法由{tp(YRS)}年提出，经过多年发展已成为主流{tp(CN_NOUNS)}。")
        elif r < 0.42:
            texts.append(f"据最新报道，研究团队成功开发了{ca}的{cn}系统。该{cn}在{cn2}测试中表现{ca}，有望在{tp(CN_NOUNS)}产业产生重要影响。项目负责人表示，{cn}将改变{tp(CN_NOUNS)}的发展方向。")
        elif r < 0.54:
            texts.append(f"问题：为什么{ca}的{cn}能提升{cn2}性能？分析：第一，{cn}通过{tp(CN_VERBS)}和{tp(CN_VERBS)}处理{tp(CN_NOUNS)}。第二，{cn2}瓶颈在于{tp(CN_NOUNS)}效率。第三，{ca}的{cn}恰好能解决。结论：{cn}是有效的{tp(CN_VERBS)}策略。")
        elif r < 0.66:
            texts.append(f"{tp(YRS)}年，{tp(['中国','美国','日本','德国'])}科学家在{cn}领域取得重大突破。他们发现通过{tp(CN_VERBS)}{tp(CN_NOUNS)}可以显著提升{cn2}的{tp(CN_NOUNS)}。这一发现为{tp(CN_NOUNS)}的应用开辟了新方向。")
        elif r < 0.78:
            en,en2 = rng.choice(EN_NOUNS),rng.choice(EN_NOUNS)
            texts.append(f"The {en} is a {rng.choice(EN_ADJS)} {en2} approach for {rng.choice(EN_VERBS)}ing {rng.choice(EN_NOUNS)}. Studies demonstrate {rng.choice(EN_ADJS)} results on {rng.choice(EN_NOUNS)} benchmarks. Key advantage: {en} can {rng.choice(EN_VERBS)} the underlying {rng.choice(EN_NOUNS)}.")
        elif r < 0.88:
            en,en2 = rng.choice(EN_NOUNS),rng.choice(EN_NOUNS)
            texts.append(f"Q: How does {en} improve {en2} performance? A: {en} leverages {rng.choice(EN_ADJS)} {rng.choice(EN_NOUNS)} to {rng.choice(EN_VERBS)} {en2}, achieving {rng.choice(EN_ADJS)} accuracy. The core innovation is that {en} can {rng.choice(EN_VERBS)} the {rng.choice(EN_NOUNS)} more effectively.")
        else:
            en = rng.choice(EN_NOUNS)
            texts.append(f"class {en.capitalize()}Processor:\n    def __init__(self, config=None):\n        self.config = config or dict(lr=0.001, batch=32)\n    def {rng.choice(EN_VERBS)}(self, data):\n        return [x for x in data if self.score(x) > self.config.get('threshold', 0.5)]\n    def score(self, item):\n        return sum(ord(c) % 100 for c in str(item)) / 100.0")
    return texts

def load_wiki(n):
    texts = []
    wiki_path = Path("data/raw/wiki_zh.jsonl")
    if wiki_path.exists():
        with open(wiki_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line); t = obj.get("text","")
                    if 80 <= len(t) <= 400: texts.append(t)
                except: pass
                if len(texts) >= n: break
    return texts

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local_r = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_r); device = torch.device(f"cuda:{local_r}")
    seed = 42 + rank
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(seed)

    # ── Data ──
    if rank == 0: print("="*55); print("P1: 100M Pretraining (4-GPU DDP)"); print("="*55)

    t1_texts = gen_templates(50000, seed=42)
    t2_texts = load_wiki(10000)
    all_texts = t1_texts + t2_texts
    random.shuffle(all_texts)

    if rank == 0: print(f"Data: {len(all_texts):,} texts ({len(t1_texts):,} template + {len(t2_texts):,} wiki)")

    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")
    all_ids = []
    for text in all_texts:
        ids = tok.encode(text).ids; all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    tokens = torch.tensor(all_ids, dtype=torch.long)

    # ── Model ──
    cfg = ModelConfig(
        vocab_size=8192, d_model=512, n_layers=24, n_heads=8, n_kv_heads=4,
        d_ff=2048, max_seq_len=1024, rope_theta=100000.0, dropout=0.0,
        use_flash_attention=True, tie_word_embeddings=True, rms_norm_eps=1e-6,
        use_qk_norm=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_r], find_unused_parameters=False, gradient_as_bucket_view=True)
    model.train()

    if rank == 0:
        n = cfg.total_params
        print(f"Model: {n:,} params | d={cfg.d_model} L={cfg.n_layers} GQA 2:1 QK-Norm")
        print(f"Embedding: {cfg.count_parameters()['embedding']:,} ({cfg.count_parameters()['embedding']/n:.1%})")
        print(f"Unique tokens: {len(torch.unique(tokens))}")

    # ── Train/Val ──
    seq_len = 1024; bs = 8
    usable = (len(tokens) // seq_len) * seq_len
    tokens_flat = tokens[:usable].view(-1, seq_len)
    split = int(len(tokens_flat) * 0.95)
    train_t, val_t = tokens_flat[:split].flatten(), tokens_flat[split:].flatten()

    class PTDataset(torch.utils.data.Dataset):
        def __init__(self, tok_tensor, sl): self.t = tok_tensor; self.s = sl
        def __len__(self): return (len(self.t)-1)//self.s
        def __getitem__(self, i):
            start=i*self.s; end=start+self.s
            inp=self.t[start:end]; lbl=inp.clone()
            if len(inp)<self.s:
                inp=torch.cat([inp,torch.zeros(self.s-len(inp),dtype=torch.long)])
                lbl=torch.cat([lbl,torch.full((self.s-len(lbl),),0,dtype=torch.long)])
            return {"input_ids":inp,"labels":lbl}

    train_ds = PTDataset(train_t, seq_len)
    val_ds = PTDataset(val_t, seq_len)
    train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                           num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                         num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # Multi-epoch training
    max_tokens = 500_000_000  # 500M token target
    steps_per_epoch = len(train_l)
    tokens_per_step = bs * world * seq_len
    total_steps = max_tokens // tokens_per_step

    epochs = max(3, total_steps // steps_per_epoch + 1)
    total_steps = min(total_steps, epochs * steps_per_epoch)

    max_lr = 5e-4
    warmup = total_steps // 10
    decay_start = int(total_steps * 0.85)

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=False)
    gs = 0; t0 = time.time()

    if rank == 0:
        print(f"\nTraining: {epochs} epochs, ~{total_steps} steps, ~{tokens_per_step*total_steps/1e9:.1f}B tokens")
        print(f"Global batch: {bs*world}x{seq_len} = {tokens_per_step:,} tok/step")
        print(f"LR: {max_lr} WSD | Warmup={warmup} Decay@={decay_start}")
        print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    losses = []
    for epoch in range(epochs):
        train_s.set_epoch(epoch)
        for batch in train_l:
            if gs >= total_steps: break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            _, out = model(input_ids, labels=labels)
            loss = out["loss"]; loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

            if gs < warmup: lr = max_lr*(gs+1)/warmup
            elif gs < decay_start: lr = max_lr
            else:
                p = min((gs-decay_start)/max(total_steps-decay_start,1), 1.0)
                lr = max_lr*0.01+0.5*max_lr*(1.0+np.cos(np.pi*p))
            for pg in opt.param_groups: pg["lr"] = lr

            losses.append(loss.item()); gs += 1

            if rank == 0 and (gs <= 20 or gs % 200 == 0):
                elapsed = time.time() - t0
                tps = gs * tokens_per_step / max(elapsed, 0.01)
                print(f"  step {gs:6d}/{total_steps} | loss={loss.item():.4f} ppl={np.exp(loss.item()):.0f} "
                      f"| lr={lr:.2e} | Δ{gn:.1f} | {tps/1000:.0f}K tok/s | {gs*tokens_per_step/1e9:.2f}B tok")

            # Eval every 500 steps
            if gs % 500 == 0 and rank == 0:
                model.eval(); et = []
                with torch.no_grad():
                    for ei, eb in enumerate(val_l):
                        if ei >= 10: break
                        _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                        et.append(eo["loss"].item())
                ep = np.exp(np.mean(et))
                print(f"  >>> VAL PPL @ {gs}: {ep:.0f} <<<")
                model.train()

            # Save every 1000 steps
            if gs % 1000 == 0 and rank == 0:
                ckpt_dir = Path("checkpoints/p1_100m")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(ckpt_dir / f"step_{gs}.pt", model.module, opt, None, step=gs, epoch=epoch, config={})
                print(f"  💾 checkpoint: step_{gs}.pt")

            if np.isnan(loss.item()):
                if rank == 0: print(f"  ❌ NaN at step {gs}!")
                break

        if gs >= total_steps: break

    # ── Final ──
    if rank == 0:
        elapsed = time.time() - t0
        model.eval(); et = []
        with torch.no_grad():
            for ei, eb in enumerate(val_l):
                if ei >= 20: break
                _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                et.append(eo["loss"].item())
        val_ppl = np.exp(np.mean(et))

        ckpt_dir = Path("checkpoints/p1_100m")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_dir / "final.pt", model.module, opt, None, step=gs, epoch=0, config={"phase":"P1_100m"})

        with open(ckpt_dir / "summary.json", "w") as f:
            json.dump({"total_params":cfg.total_params,"d_model":cfg.d_model,"n_layers":cfg.n_layers,
                       "total_steps":gs,"total_tokens":gs*tokens_per_step,"val_ppl":float(val_ppl),
                       "time_hr":elapsed/3600,"throughput_ktok":gs*tokens_per_step/elapsed/1000}, f, indent=2)

        print(f"\n{'='*55}\nP1 Complete! {elapsed/3600:.1f}hr | {gs*tokens_per_step/1e9:.2f}B tokens")
        print(f"VAL PPL: {val_ppl:.0f} | {gs*tokens_per_step/elapsed/1000:.0f}K tok/s")
        print(f"Checkpoint: {ckpt_dir / 'final.pt'}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
