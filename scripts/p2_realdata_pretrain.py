#!/usr/bin/env python
"""P2 Plan B: MiniMind Real Data Pretraining — 100M model, 4-GPU DDP.

Data: MiniMind pretrain_t2t_mini.jsonl (1.2GB, 1.27M docs, ~10B tokens).
This is REAL Chinese web text — clean, deduped, diverse.
Same data that trained MiniMind's 26M/108M models.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/p2_realdata_pretrain.py
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
from src.utils.checkpoint import save_checkpoint

random.seed(42)

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

    if rank == 0:
        print("="*55)
        print(f"P2 Plan B: Real Data Pretraining ({world}×RTX3090 DDP)")
        print("="*55)

    # ── Load MiniMind pretrain data ──
    DATA_PATH = "/wuzhou/pentafleet/b23113_/minimind-master/dataset/pretrain_t2t_mini.jsonl"

    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    if rank == 0: print(f"Loading {DATA_PATH}...")

    all_ids = []
    bos_id, eos_id = 1, 2
    line_count = 0
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text or len(text) < 20:
                    continue
                ids = tok.encode(text).ids
                all_ids.append(bos_id)
                all_ids.extend(ids)
                all_ids.append(eos_id)
                line_count += 1
                if line_count >= 500000:  # Sample 500K docs for fast test (~4-5B tokens)
                    break
            except:
                continue

    tokens = torch.tensor(all_ids, dtype=torch.long)

    if rank == 0:
        unique = len(torch.unique(tokens))
        print(f"  Docs: {line_count:,} | Tokens: {len(tokens):,}")
        print(f"  Unique tokens: {unique}/8192 ({unique/8192:.1%})")
        # Show a sample
        sample = tok.decode(tokens[100:200].tolist(), skip_special_tokens=True)
        print(f"  Sample text: {sample[:80]}...")

    # ── Model: 100M ──
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

    # ── Train/Val ──
    seq_len = 1024; bs = 8
    usable = (len(tokens) // seq_len) * seq_len
    tokens_flat = tokens[:usable].view(-1, seq_len)
    split = int(len(tokens_flat) * 0.95)

    class PTDataset(torch.utils.data.Dataset):
        def __init__(self, tok_tensor, sl): self.t = tok_tensor; self.s = sl
        def __len__(self): return len(self.t)
        def __getitem__(self, i):
            inp = self.t[i]; lbl = inp.clone()
            return {"input_ids": inp, "labels": lbl}

    train_ds = PTDataset(tokens_flat[:split], seq_len)
    val_ds = PTDataset(tokens_flat[split:], seq_len)
    train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                           num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                         num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    tokens_per_step = bs * world * seq_len
    total_steps = len(train_l) * 3  # 3 epochs on this data

    max_lr = 5e-4; warmup = total_steps // 10; decay_start = int(total_steps * 0.85)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    gs = 0; t0 = time.time()

    if rank == 0:
        print(f"\nTraining: 3 epochs, ~{total_steps} steps")
        print(f"Tokens: ~{tokens_per_step*total_steps/1e9:.1f}B | Batch: {bs*world}x{seq_len}={tokens_per_step:,}")
        print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    for epoch in range(3):
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
            gs += 1

            if rank == 0 and (gs <= 20 or gs % 100 == 0):
                elapsed = time.time() - t0
                tps = gs * tokens_per_step / max(elapsed, 0.01)
                print(f"  step {gs:6d}/{total_steps} | loss={loss.item():.4f} ppl={np.exp(loss.item()):.0f} "
                      f"| {tps/1000:.0f}K tok/s | {gs*tokens_per_step/1e9:.2f}B tok")

            if gs % 500 == 0 and rank == 0:
                model.eval(); et = []
                with torch.no_grad():
                    for ei, eb in enumerate(val_l):
                        if ei >= 12: break
                        _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                        et.append(eo["loss"].item())
                print(f"  >>> VAL PPL @ {gs}: {np.exp(np.mean(et)):.0f} <<<")
                model.train()

            if gs % 3000 == 0 and rank == 0:
                ckpt_dir = Path("checkpoints/p2_realdata")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(ckpt_dir / f"step_{gs}.pt", model.module, opt, None, step=gs, epoch=epoch, config={})

    if rank == 0:
        elapsed = time.time() - t0
        model.eval(); et = []
        with torch.no_grad():
            for ei, eb in enumerate(val_l):
                if ei >= 20: break
                _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                et.append(eo["loss"].item())
        val_ppl = np.exp(np.mean(et))

        ckpt_dir = Path("checkpoints/p2_realdata")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_dir / "final.pt", model.module, opt, None, step=gs, epoch=0,
                        config={"plan":"B","data":"pretrain_t2t_mini","tokens":gs*tokens_per_step,"val_ppl":float(val_ppl)})

        print(f"\n{'='*55}\nP2 Plan B Complete! {elapsed/3600:.1f}hr")
        print(f"VAL PPL: {val_ppl:.0f} | {gs*tokens_per_step/elapsed/1000:.0f}K tok/s")
        print(f"Checkpoint: {ckpt_dir / 'final.pt'}")

        # ── Quick generative test ──
        print(f"\nGenerative Test:")
        model.eval()
        for prompt in ["人工智能是","北京是中国的","春天来了","中国最大的城市是"]:
            ids = tok.encode(prompt).ids
            pid = torch.tensor([[1]+ids], device=device)
            out_tokens = []
            for tid, is_done in model.module.generate_stream(pid, max_new_tokens=30, temperature=0.8, top_k=35, top_p=0.9, eos_token_id=2):
                out_tokens.append(tid)
                if is_done: break
            print(f"  {prompt} {tok.decode(out_tokens, skip_special_tokens=True)[:80]}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
