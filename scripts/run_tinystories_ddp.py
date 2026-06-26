#!/usr/bin/env python
"""TinyStories 4-GPU DDP benchmark — verify 14M model pipeline health."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np, time
from datetime import datetime
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local_r = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_r); device = torch.device(f"cuda:{local_r}")
    seed = 42 + rank
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    if rank == 0:
        print("=" * 55)
        print(f"TinyStories 4-GPU DDP Benchmark ({world}×RTX3090)")
        print("=" * 55)

    # ── Data ──
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    if rank == 0: print("Loading TinyStories (200K stories)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10000)

    texts = []
    for i, s in enumerate(ds):
        texts.append(s["text"].strip())
        if i % 50000 == 0 and rank == 0: print(f"  {i}/200000...", end="\r")
        if len(texts) >= 200_000: break

    # Batch tokenize
    if rank == 0: print("\nTokenizing...")
    tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    all_ids = []
    for i in range(0, len(texts), 10000):
        batch = texts[i:i+10000]
        enc = tok(batch, add_special_tokens=True, truncation=True, max_length=512,
                   return_attention_mask=False, return_token_type_ids=False)
        for ids in enc["input_ids"]: all_ids.extend(ids)
    tokens = torch.tensor(all_ids, dtype=torch.long)

    # ── Model ~14M ──
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, d_model=192, n_layers=7, n_heads=6, n_kv_heads=6,
        d_ff=512, max_seq_len=256, dropout=0.0, use_flash_attention=True,
        tie_word_embeddings=True, rms_norm_eps=1e-6,
        pad_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
    )
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_r], find_unused_parameters=False, gradient_as_bucket_view=True)
    model.train()

    # ── Train/Val split ──
    seq_len = 256; bs = 16  # per GPU, global=64
    total_seqs = len(tokens) // seq_len
    usable = total_seqs * seq_len
    tokens_flat = tokens[:usable].view(-1, seq_len)
    split = int(total_seqs * 0.95)
    train_t, val_t = tokens_flat[:split], tokens_flat[split:]

    train_ds = PretrainDataset(train_t.flatten(), seq_len=seq_len)
    val_ds = PretrainDataset(val_t.flatten(), seq_len=seq_len)
    train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                           num_workers=2, pin_memory=True, prefetch_factor=2,
                                           persistent_workers=True)
    val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                         num_workers=2, pin_memory=True, prefetch_factor=2,
                                         persistent_workers=True)

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"  Vocab: {tok.vocab_size}, Tokens: {len(all_ids):,}")
        print(f"  Model: {n:,} params | d={cfg.d_model} L={cfg.n_layers} d_ff={cfg.d_ff}")
        print(f"  Global batch: {bs*world}x{seq_len}")

    # ── Train ──
    epochs = 3; max_lr = 5e-4
    total_steps = len(train_l) * epochs
    warmup = total_steps // 10; decay_start = int(total_steps * 0.85)

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95))
    gs = 0; t0 = time.time()

    for epoch in range(epochs):
        train_s.set_epoch(epoch)
        for batch in train_l:
            if gs >= total_steps: break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            _, out = model(input_ids, labels=labels)
            loss = out["loss"]; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

            # WSD LR
            if gs < warmup: lr = max_lr * (gs+1)/warmup
            elif gs < decay_start: lr = max_lr
            else:
                p = min((gs-decay_start)/max(total_steps-decay_start,1), 1.0)
                lr = max_lr*0.01 + 0.5*max_lr*(1.0+np.cos(np.pi*p))
            for pg in opt.param_groups: pg["lr"] = lr

            gs += 1
            if rank == 0 and (gs <= 20 or gs % 100 == 0):
                elapsed = time.time() - t0
                tps = gs * bs * world * seq_len / max(elapsed, 0.01)
                print(f"  step {gs:5d}/{total_steps} | loss={loss.item():.4f} "
                      f"ppl={np.exp(loss.item()):.0f} | {tps/1000:.0f}K tok/s")

            if gs % 300 == 0 and rank == 0:
                model.eval(); et = []
                with torch.no_grad():
                    for ei, eb in enumerate(val_l):
                        if ei >= 10: break
                        eid = eb["input_ids"].to(device); ela = eb["labels"].to(device)
                        _, eo = model(eid, labels=ela); et.append(eo["loss"].item())
                print(f"  >>> VAL PPL: {np.exp(np.mean(et)):.0f} <<<")
                model.train()

        if gs >= total_steps: break

    # ── Final ──
    if rank == 0:
        model.eval(); et = []
        with torch.no_grad():
            for ei, eb in enumerate(val_l):
                if ei >= 20: break
                eid = eb["input_ids"].to(device); ela = eb["labels"].to(device)
                _, eo = model(eid, labels=ela); et.append(eo["loss"].item())
        val_ppl = np.exp(np.mean(et))
        elapsed = time.time() - t0

        print(f"\n{'='*55}")
        print(f"TinyStories DDP Results")
        print(f"{'='*55}")
        print(f"  GPUs: {world} | Steps: {gs} | Time: {elapsed/60:.1f}min")
        print(f"  VAL PPL: {val_ppl:.0f}")

        if val_ppl < 30:
            print(f"  ✅ PIPELINE HEALTHY — PPL within TinyStories range (15-25)")
        elif val_ppl < 60:
            print(f"  ⚠️  MARGINAL — learning but above baseline")
        else:
            print(f"  ❌ PIPELINE ISSUE — check before changing Chinese data")

        # Generate
        prompt = "Once upon a time, there was a little"
        pid = torch.tensor([tok.encode(prompt)], device=device)
        with torch.no_grad():
            full_ids, _ = model.module.generate(pid, max_new_tokens=60, temperature=0.8, top_k=30, top_p=0.9,
                                                 eos_token_id=tok.eos_token_id)
        print(f"\n  Prompt: {prompt}")
        print(f"  Story:  {tok.decode(full_ids[0].tolist(), skip_special_tokens=True)[:250]}")
        print(f"\n✅ Done!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
