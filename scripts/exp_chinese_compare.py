#!/usr/bin/env python
"""Chinese data source control experiment: Wiki vs FineWeb-Edu.

Single-variable test: same 14M model + same tokenizer + same training.
Only difference: data source.
Expected: isolate whether PPL=2200 is wiki-specific or general Chinese text problem.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/exp_chinese_compare.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np, json, time, random

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
    random.seed(seed)

    # ── Shared config ──
    seq_len = 256; bs = 16
    max_lr = 5e-4; epochs = 5; warmup_pct = 0.10; decay_pct = 0.85

    # ── Load Chinese tokenizer ──
    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    def tokenize(texts):
        all_ids = []
        for t in texts:
            ids = tok.encode(t).ids
            all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
        return torch.tensor(all_ids, dtype=torch.long)

    def load_wiki(n):
        texts = []
        with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line); text = obj.get("text", "")
                if 80 <= len(text) <= 300: texts.append(text)
                if len(texts) >= n: break
        return texts

    def load_fineweb_zh(n):
        """Try to load Chinese FineWeb-Edu from hf-mirror."""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        texts = []
        for s in ds:
            text = s.get("text", "")
            # Filter for Chinese content
            cjk = sum(1 for c in text if '一' <= c <= '鿿')
            if cjk > len(text) * 0.3 and 80 <= len(text) <= 400:
                texts.append(text)
            if len(texts) >= n: break
        return texts

    def load_chinese_wiki_raw(n):
        """Load raw Chinese wiki paragraphs (no length filter, real distribution)."""
        texts = []
        with open("data/raw/wiki_zh.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line); text = obj.get("text", "")
                if len(text) >= 50: texts.append(text)
                if len(texts) >= n: break
        return texts

    results = {}

    for name, loader, n_samples in [
        ("wiki_curated", load_wiki, 5000),
        ("wiki_raw", load_chinese_wiki_raw, 5000),
    ]:
        if rank == 0: print(f"\n{'='*50}\n[{name}] {n_samples} samples\n{'='*50}")

        texts = loader(n_samples)
        tokens = tokenize(texts)
        total_tokens = len(tokens)

        if rank == 0:
            unique = len(torch.unique(tokens))
            print(f"  Texts: {len(texts):,}, Tokens: {total_tokens:,}, Unique: {unique}/8192 ({unique/8192:.1%})")

        # Model
        cfg = ModelConfig(
            vocab_size=tok.get_vocab_size(), d_model=384, n_layers=6,
            n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=seq_len,
            dropout=0.0, use_flash_attention=True, tie_word_embeddings=True,
            rms_norm_eps=1e-6, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        model = Transformer(cfg).to(device)
        model = DDP(model, device_ids=[local_r], find_unused_parameters=False,
                    gradient_as_bucket_view=True)
        model.train()

        # Train/Val split
        usable = (len(tokens) // seq_len) * seq_len
        tokens_flat = tokens[:usable].view(-1, seq_len)
        split = int(len(tokens_flat) * 0.9)
        train_t = tokens_flat[:split]; val_t = tokens_flat[split:]

        train_ds = PretrainDataset(train_t.flatten(), seq_len=seq_len)
        val_ds = PretrainDataset(val_t.flatten(), seq_len=seq_len)
        train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
        train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                               num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                             num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

        total_steps = len(train_l) * epochs
        warmup = int(total_steps * warmup_pct)
        decay_start = int(total_steps * decay_pct)

        opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95))
        gs = 0; t0 = time.time()

        for epoch in range(epochs):
            train_s.set_epoch(epoch)
            for batch in train_l:
                if gs >= total_steps: break
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                _, out = model(input_ids, labels=labels); loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()

                if gs < warmup: lr = max_lr*(gs+1)/warmup
                elif gs < decay_start: lr = max_lr
                else:
                    p = min((gs-decay_start)/max(total_steps-decay_start,1), 1.0)
                    lr = max_lr*0.01+0.5*max_lr*(1.0+np.cos(np.pi*p))
                for pg in opt.param_groups: pg["lr"] = lr
                gs += 1

                if rank == 0 and (gs <= 10 or gs % 100 == 0):
                    print(f"  step {gs:4d}/{total_steps} loss={loss.item():.4f} ppl={np.exp(loss.item()):.0f} lr={lr:.2e}")

                if gs % 300 == 0 and rank == 0:
                    model.eval(); et = []
                    with torch.no_grad():
                        for ei, eb in enumerate(val_l):
                            if ei >= 10: break
                            _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                            et.append(eo["loss"].item())
                    vppl = np.exp(np.mean(et))
                    print(f"  >>> [{name}] VAL PPL @ {gs}: {vppl:.0f} <<<")
                    model.train()

        # Final eval
        if rank == 0:
            model.eval(); et = []
            with torch.no_grad():
                for ei, eb in enumerate(val_l):
                    if ei >= 20: break
                    _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                    et.append(eo["loss"].item())
            val_ppl = np.exp(np.mean(et))
            elapsed = time.time() - t0
            results[name] = val_ppl
            print(f"  [{name}] FINAL PPL: {val_ppl:.0f} | {elapsed/60:.1f}min | {gs} steps")

    # ── Summary ──
    if rank == 0:
        print(f"\n{'='*55}")
        print("Chinese Data Source Comparison")
        print(f"{'='*55}")
        for n, p in results.items():
            print(f"  {n:<25} PPL={p:.0f}")
        print(f"\n  Baseline: Chinese Wiki (full, 146K docs) → PPL 2200")
        print(f"  Baseline: TinyStories (English) → PPL 6")
        print(f"\n  If wiki_curated PPL << 2200: data diversity is the issue")
        print(f"  If wiki_curated PPL ~ 2200: Chinese text fundamentally harder for 14M")
        print(f"✅ Done!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
