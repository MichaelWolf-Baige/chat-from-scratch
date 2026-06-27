#!/usr/bin/env python
"""Step 1: Train 100M model on any JSONL data → save checkpoint. No generation.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=9 scripts/train_pretrain.py \
        --data data/distill_merged.jsonl --output checkpoints/p3_ours/final.pt --epochs 2
"""
import os, sys, argparse, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np, random
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.utils.checkpoint import save_checkpoint

random.seed(42)

def load_texts(path, max_docs=None):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
                if len(t) >= 30: texts.append(t)
            except: pass
            if max_docs and len(texts) >= max_docs: break
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="JSONL data path")
    parser.add_argument("-o", "--output", default="checkpoints/train_output/final.pt")
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("--max_docs", type=int, default=50000)
    parser.add_argument("--bs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    # DDP
    dist.init_process_group(backend="nccl"); rank = dist.get_rank(); world = dist.get_world_size()
    local_r = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_r)
    device = torch.device(f"cuda:{local_r}")
    torch.manual_seed(42+rank); torch.cuda.manual_seed_all(42+rank)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(42)

    from tokenizers import Tokenizer as HFTok; tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    # Load + tokenize
    texts = load_texts(args.data, args.max_docs)
    random.shuffle(texts)
    all_ids = []
    for t in texts:
        ids = tok.encode(t).ids; all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    tokens = torch.tensor(all_ids, dtype=torch.long)

    if rank == 0:
        print(f"\n{'='*55}\nTRAINING: {args.data}\n  {len(texts):,} texts → {len(tokens):,} tokens\n{'='*55}")

    # Model
    cfg = ModelConfig(vocab_size=8192, d_model=512, n_layers=24, n_heads=8, n_kv_heads=4,
                      d_ff=2048, max_seq_len=1024, rope_theta=100000.0, dropout=0.0,
                      use_flash_attention=True, tie_word_embeddings=True, rms_norm_eps=1e-6,
                      use_qk_norm=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_r], find_unused_parameters=False, gradient_as_bucket_view=True)
    model.train()

    # DataLoader
    seq_len = 1024; bs = args.bs
    usable = (len(tokens) // seq_len) * seq_len
    tokens_flat = tokens[:usable].view(-1, seq_len)
    split = int(len(tokens_flat) * 0.95)

    class DS(torch.utils.data.Dataset):
        def __init__(self, t, sl): self.t = t; self.s = sl
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return {"input_ids": self.t[i], "labels": self.t[i].clone()}

    train_ds = DS(tokens_flat[:split], seq_len)
    val_ds = DS(tokens_flat[split:], seq_len)
    train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                           num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                         num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # Training
    tps = bs * world * seq_len
    total_steps = len(train_l) * args.epochs
    max_lr = args.lr; warmup = total_steps // 10; decay_start = int(total_steps * 0.85)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    gs = 0; t0 = time.time()

    if rank == 0:
        n = cfg.total_params
        print(f"  {n:,} params | {world} GPUs | {args.epochs} epochs | {total_steps} steps")
        print(f"  Global batch: {bs*world}x{seq_len} | ~{total_steps*tps/1e9:.2f}B tokens | LR={max_lr}")

    for epoch in range(args.epochs):
        train_s.set_epoch(epoch)
        for batch in train_l:
            if gs >= total_steps: break
            iid = batch["input_ids"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)
            _, out = model(iid, labels=lbl)
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

            if rank == 0 and (gs <= 10 or gs % 50 == 0):
                el = time.time() - t0
                print(f"  step {gs:5d}/{total_steps} | loss={loss.item():.4f} ppl={np.exp(loss.item()):.0f} "
                      f"| {gs*tps/max(el,0.01)/1000:.0f}K tok/s")

    # ── Final eval + save (rank 0 only) ──
    dist.barrier()
    if rank == 0:
        model.eval(); et = []
        with torch.no_grad():
            for ei, eb in enumerate(val_l):
                if ei >= 15: break
                _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                et.append(eo["loss"].item())
        val_ppl = np.exp(np.mean(et))
        elapsed = time.time() - t0

        # Save
        ckpt_path = Path(args.output); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model_state = model.module.state_dict()
        torch.save({"model": model_state, "val_ppl": float(val_ppl),
                     "config": {"n_texts": len(texts), "n_tokens": len(tokens),
                                "total_steps": gs, "epochs": args.epochs}},
                    ckpt_path)

        print(f"\n{'='*55}\nDONE: {args.output}")
        print(f"  VAL PPL: {val_ppl:.0f} | {elapsed/60:.1f}min | {gs*tps/elapsed/1000:.0f}K tok/s")
        print(f"{'='*55}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
