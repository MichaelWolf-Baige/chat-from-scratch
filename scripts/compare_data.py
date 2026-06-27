#!/usr/bin/env python
"""Controlled data comparison: our distilled data vs MiniMind data.

Same 100M model, same training config, same token count.
Only variable: data source.

A: Our distilled data (~80MB, distilled_merged.jsonl)
B: MiniMind data (sampled to equal token count for fair comparison)
C: MiniMind full mini data (1.2GB, upper bound)

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/compare_data.py
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

random.seed(42)

def load_jsonl(path, max_docs=None):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if len(text) >= 30:
                    texts.append(text)
            except:
                pass
            if max_docs and len(texts) >= max_docs:
                break
    return texts

def tokenize_with_tokenizer(texts, tok):
    all_ids = []
    for t in texts:
        ids = tok.encode(t).ids; all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    return torch.tensor(all_ids, dtype=torch.long)

def train_and_eval(name, tokens, world, rank, local_r, device, max_steps=None, max_tokens=None):
    seq_len = 1024; bs = 12
    usable = (len(tokens) // seq_len) * seq_len
    tokens_flat = tokens[:usable].view(-1, seq_len)
    split = int(len(tokens_flat) * 0.95)

    class PTDataset(torch.utils.data.Dataset):
        def __init__(self, t, sl): self.t=t; self.s=sl
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return {"input_ids": self.t[i], "labels": self.t[i].clone()}

    train_ds = PTDataset(tokens_flat[:split], seq_len)
    val_ds = PTDataset(tokens_flat[split:], seq_len)
    train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                           num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                         num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # Model
    cfg = ModelConfig(vocab_size=8192, d_model=512, n_layers=24, n_heads=8, n_kv_heads=4,
                      d_ff=2048, max_seq_len=1024, rope_theta=100000.0, dropout=0.0,
                      use_flash_attention=True, tie_word_embeddings=True, rms_norm_eps=1e-6,
                      use_qk_norm=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_r], find_unused_parameters=False, gradient_as_bucket_view=True)
    model.train()

    tokens_per_step = bs * world * seq_len
    if max_steps is None:
        max_steps = len(train_l) * 2  # 2 epochs, fast
    total_steps = max_steps  # use 2 epochs

    max_lr = 5e-4; warmup = total_steps // 10; decay_start = int(total_steps * 0.85)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    gs = 0; t0 = time.time()

    if rank == 0:
        print(f"\n[{name}] {len(tokens):,} tokens | {total_steps} steps | {total_steps*tokens_per_step/1e9:.1f}B tok | 2 epochs")

    losses = []
    for epoch in range(2):  # 2 epochs for fast comparison
        train_s.set_epoch(epoch)
        for batch in train_l:
            if gs >= total_steps: break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            _, out = model(input_ids, labels=labels)
            loss = out["loss"]; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

            if gs < warmup: lr = max_lr*(gs+1)/warmup
            elif gs < decay_start: lr = max_lr
            else:
                p = min((gs-decay_start)/max(total_steps-decay_start,1), 1.0)
                lr = max_lr*0.01+0.5*max_lr*(1.0+np.cos(np.pi*p))
            for pg in opt.param_groups: pg["lr"] = lr
            losses.append(loss.item()); gs += 1

            if rank == 0 and (gs <= 10 or gs % 100 == 0):
                elapsed = time.time() - t0
                tps = gs * tokens_per_step / max(elapsed, 0.01)
                print(f"  [{name}] step {gs:5d}/{total_steps} | loss={loss.item():.4f} "
                      f"ppl={np.exp(loss.item()):.0f} | {tps/1000:.0f}K tok/s")

        if gs >= total_steps: break

    # Final eval + save
    if rank == 0:
        model.eval(); et = []
        with torch.no_grad():
            for ei, eb in enumerate(val_l):
                if ei >= 15: break
                _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                et.append(eo["loss"].item())
        val_ppl = np.exp(np.mean(et))
        elapsed = time.time() - t0
        print(f"  [{name}] VAL PPL={val_ppl:.0f} | {elapsed/60:.1f}min | {gs*tokens_per_step/1e9:.1f}B tokens")
        return val_ppl, model.module
    return None, None


def generative_test(model, tok, name):
    prompts = ["人工智能是","北京是中国的","春天来了，","什么是机器学习？","今天天气",]
    print(f"\n[{name}] Generative Test:")
    for prompt in prompts:
        ids = tok.encode(prompt).ids
        pid = torch.tensor([[1]+ids], device="cuda")
        out_tokens = []
        for tid, is_done in model.generate_stream(pid, max_new_tokens=30, temperature=0.8, top_k=35, top_p=0.9, eos_token_id=2):
            out_tokens.append(tid)
            if is_done: break
        print(f"  {prompt} {tok.decode(out_tokens, skip_special_tokens=True)[:80]}")


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local_r = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_r); device = torch.device(f"cuda:{local_r}")
    seed = 42 + rank
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(seed)

    if rank == 0: print("="*55); print("DATA COMPARISON: Our Distilled vs MiniMind"); print("="*55)

    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    # ── Load data ──
    if rank == 0: print("Loading data...")

    # A: Our distilled (all of it — 80MB)
    our_texts = load_jsonl("data/distill_merged.jsonl")
    our_tokens = tokenize_with_tokenizer(our_texts, tok)

    # B: MiniMind — sample FIRST, then tokenize only the sample
    # This avoids tokenizing 1.2GB of data
    mm_texts = load_jsonl("/wuzhou/pentafleet/b23113_/minimind-master/dataset/pretrain_t2t_mini.jsonl",
                          max_docs=len(our_texts))  # same count as ours
    mm_tokens = tokenize_with_tokenizer(mm_texts, tok)

    if rank == 0:
        print(f"  A (Our distilled): {len(our_texts):,} texts → {len(our_tokens):,} tokens")
        print(f"  B (MiniMind eq):  {min(len(mm_texts), len(our_texts)):,} texts → {len(mm_tokens):,} tokens (sampled to match)")
        unique_a = len(torch.unique(our_tokens[:min(100000, len(our_tokens))]))
        unique_b = len(torch.unique(mm_tokens[:min(100000, len(mm_tokens))]))
        print(f"  Token diversity: Ours={unique_a} unique | MM={unique_b} unique (sample of 100K)")

    # ── Train A: Our data ──
    ppl_a, model_a = train_and_eval("OURS", our_tokens, world, rank, local_r, device)

    # ── Train B: MiniMind equal ──
    ppl_b, model_b = train_and_eval("MINIMIND", mm_tokens, world, rank, local_r, device)

    # ── Results ──
    if rank == 0:
        print(f"\n{'='*55}")
        print(f"RESULTS")
        print(f"{'='*55}")
        print(f"  A (Our distilled):  VAL PPL = {ppl_a:.0f}")
        print(f"  B (MiniMind equal): VAL PPL = {ppl_b:.0f}")
        print(f"  C (MiniMind full):  VAL PPL = 7 (from Plan B, 500K docs)")

        generative_test(model_a, tok, "A: Our Distilled")
        generative_test(model_b, tok, "B: MiniMind (equal tokens)")

        # Save models
        p_a = Path("checkpoints/compare_ours/final.pt"); p_a.parent.mkdir(parents=True, exist_ok=True)
        p_b = Path("checkpoints/compare_minimind/final.pt"); p_b.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model_a.state_dict()}, p_a)
        torch.save({"model": model_b.state_dict()}, p_b)

        print(f"\n✅ Models saved: {p_a}, {p_b}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
