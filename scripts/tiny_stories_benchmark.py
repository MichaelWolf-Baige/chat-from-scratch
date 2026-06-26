#!/usr/bin/env python
"""TinyStories benchmark: reproduce known 14M model PPL ~15-25.

If PPL < 30: pipeline is healthy, the 2200 PPL on Chinese wiki is a data/hyperparameter mismatch.
If PPL > 100: pipeline or hyperparameters have issues — fix before changing data.

TinyStories paper (Eldan & Li 2023): 14M Llama-style model achieves test PPL 15-25.

Usage:
    python scripts/tiny_stories_benchmark.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.model.position import apply_rotary_emb
from src.data.tokenizer_utils import load_tokenizer

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Download TinyStories via hf-mirror
# ═══════════════════════════════════════════════════════════════════

def download_tiny_stories(output_dir="data/tiny_stories/"):
    """Download TinyStories from hf-mirror. Returns list of texts."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    cache_file = output_path / "tiny_stories_train.jsonl"
    if cache_file.exists():
        texts = []
        with open(cache_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
        print(f"  Loaded {len(texts):,} texts from cache")
        return texts

    print("  Downloading TinyStories from hf-mirror...")
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)

    texts = []
    with open(cache_file, "w", encoding="utf-8") as f:
        for sample in ds:
            text = sample["story"].strip()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            texts.append(text)

    print(f"  Downloaded {len(texts):,} stories, saved to {cache_file}")
    return texts


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Tokenize with GPT-2 tokenizer (matching TinyStories paper)
# ═══════════════════════════════════════════════════════════════════

def tokenize_tiny_stories(texts, seq_len=512):
    """Tokenize using GPT-2 tokenizer (matches TinyStories paper)."""
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token  # GPT-2 has no pad token

    # Tokenize in chunks to avoid memory issues
    all_ids = []
    for text in texts:
        ids = tok.encode(text)
        all_ids.append(tok.bos_token_id or tok.eos_token_id)
        all_ids.extend(ids)
        all_ids.append(tok.eos_token_id)

    tokens = torch.tensor(all_ids, dtype=torch.long)
    return tokens, tok


# ═══════════════════════════════════════════════════════════════════
# STEP 3: Train 14M model matching TinyStories paper
# ═══════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 55)
    print("TinyStories Benchmark: 14M Model")
    print("=" * 55)

    # ── Data ──
    texts = download_tiny_stories()
    tokens, gpt2_tok = tokenize_tiny_stories(texts, seq_len=512)
    vocab_size = gpt2_tok.vocab_size

    # Split: 90% train, 10% validation
    split = int(len(tokens) * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    # Count stats
    total_tokens = len(tokens)
    print(f"  GPT-2 vocab: {vocab_size}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Train: {len(train_tokens):,}, Val: {len(val_tokens):,}")
    print(f"  Sample story: {texts[0][:150]}...")

    # ── Model: 14M, GPT-2 tokenizer compatible ──
    # TinyStories paper uses Llama-style with GPT-2 tokenizer (50257 vocab)
    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=288,          # smaller to keep ~14M total
        n_layers=6,
        n_heads=6,
        n_kv_heads=6,
        d_ff=768,
        max_seq_len=512,
        rope_theta=10000.0,
        dropout=0.0,
        use_flash_attention=True,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        pad_token_id=gpt2_tok.pad_token_id or gpt2_tok.eos_token_id,
        bos_token_id=gpt2_tok.bos_token_id or gpt2_tok.eos_token_id,
        eos_token_id=gpt2_tok.eos_token_id,
    )
    model = Transformer(cfg).to(device)
    n_params = cfg.total_params
    emb_ratio = cfg.count_parameters()["embedding"] / n_params
    print(f"\n  Model: {n_params:,} params")
    print(f"  Embedding: {cfg.count_parameters()['embedding']:,} ({emb_ratio:.0%} of total)")
    print(f"  FFN: {cfg.count_parameters()['ffn']:,} | Attn: {cfg.count_parameters()['attention']:,}")

    # ── Training setup ──
    seq_len = 256  # shorter context = faster benchmarks
    bs = 32
    epochs = 3
    total_seqs = len(train_tokens) // seq_len
    steps_per_epoch = total_seqs // bs
    total_steps = steps_per_epoch * epochs
    lr = 5e-4
    warmup = total_steps // 10
    decay_start = int(total_steps * 0.85)

    # Truncate to seq_len boundary
    usable = (len(train_tokens) // seq_len) * seq_len
    train_t = train_tokens[:usable].view(-1, seq_len)[:total_seqs]

    usable_v = (len(val_tokens) // seq_len) * seq_len
    val_t = val_tokens[:usable_v].view(-1, seq_len)[:min(total_seqs // 10, usable_v // seq_len)]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()

    print(f"\n  Training: {epochs} epochs x {steps_per_epoch} steps = {total_steps} total")
    print(f"  LR: {lr}, WSD (warmup={warmup}, decay_start={decay_start})")
    print(f"  Tokens: {total_steps * bs * seq_len:,}")
    print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")

    losses = []
    global_step = 0
    tok_start = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(total_seqs)
        for i in range(0, total_seqs - bs, bs):
            idx = perm[i:i + bs]
            batch = train_t[idx].to(device)

            _, out = model(batch, labels=batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            # WSD LR
            if global_step < warmup:
                lr_now = lr * (global_step + 1) / warmup
            elif global_step < decay_start:
                lr_now = lr
            else:
                p = min((global_step - decay_start) / max(total_steps - decay_start, 1), 1.0)
                lr_now = lr * 0.01 + 0.5 * lr * (1.0 + np.cos(np.pi * p))
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            losses.append(loss.item())
            global_step += 1

            if global_step <= 20 or global_step % 100 == 0:
                elapsed = time.time() - tok_start
                tps = global_step * bs * seq_len / max(elapsed, 0.01)
                print(f"  step {global_step:5d} | loss={loss.item():.4f} | "
                      f"ppl={np.exp(loss.item()):.0f} | lr={lr_now:.2e} | {tps/1000:.0f}K tok/s")

            # Eval every 400 steps
            if global_step % 400 == 0:
                model.eval()
                et, en_ = 0.0, 0
                with torch.no_grad():
                    for ei in range(0, min(len(val_t) - bs, bs * 8), bs):
                        vb = val_t[ei:ei + bs].to(device)
                        _, eo = model(vb, labels=vb)
                        et += eo["loss"].item()
                        en_ += 1
                ep = np.exp(et / max(en_, 1))
                print(f"  >>> VAL PPL: {ep:.0f} <<<")
                model.train()

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    # ── Final eval ──
    model.eval()
    et, en_ = 0.0, 0
    with torch.no_grad():
        for ei in range(0, min(len(val_t) - bs, bs * 15), bs):
            vb = val_t[ei:ei + bs].to(device)
            _, eo = model(vb, labels=vb)
            et += eo["loss"].item()
            en_ += 1
    final_ppl = np.exp(et / max(en_, 1))
    elapsed = time.time() - tok_start

    print(f"\n{'='*55}")
    print(f"TinyStories Benchmark Results")
    print(f"{'='*55}")
    print(f"  Model: {n_params:,} params")
    print(f"  Steps: {global_step} | Tokens: {global_step * bs * seq_len:,}")
    print(f"  Time: {elapsed/60:.1f}min | {global_step*bs*seq_len/elapsed:.0f} tok/s")
    print(f"  Train loss: {losses[0]:.2f} -> {losses[-1]:.4f}")
    print(f"  Train PPL: {np.exp(losses[0]):.0f} -> {np.exp(losses[-1]):.0f}")
    print(f"  VAL PPL: {final_ppl:.0f}")

    # ── Verdict ──
    print(f"\n  {'─' * 40}")
    if final_ppl < 30:
        print(f"  ✅ PIPELINE HEALTHY — PPL {final_ppl:.0f} is in the expected range (15-25)")
        print(f"     Your code, optimizer, and architecture work correctly.")
        print(f"     The 2200 PPL on Chinese wiki is a data/hyperparameter mismatch.")
        print(f"     → Focus on data quality and training hyperparameters for Chinese.")
    elif final_ppl < 60:
        print(f"  ⚠️  PIPELINE MARGINAL — PPL {final_ppl:.0f} is higher than expected (15-25)")
        print(f"     Model is learning but not as efficiently as it should.")
        print(f"     → Check: LR schedule, warmup, embedding/architecture ratios.")
    elif final_ppl < 200:
        print(f"  ⚠️  PIPELINE SUSPECT — PPL {final_ppl:.0f} is significantly above baseline")
        print(f"     Something in the training setup is suboptimal.")
        print(f"     → Check: tokenizer compatibility, batch size, gradient clipping.")
    else:
        print(f"  ❌ PIPELINE ISSUE — PPL {final_ppl:.0f} is far above baseline (15-25)")
        print(f"     A fundamental problem exists in the training pipeline.")
        print(f"     → Check: model architecture, loss computation, data encoding.")

    # ── Generate sample ──
    print(f"\n{'='*55}")
    print("Story Generation Demo")
    print(f"{'='*55}")
    prompt = "Once upon a time, there was a little"
    prompt_ids = gpt2_tok.encode(prompt)
    pid = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        full_ids, _ = model.generate(pid, max_new_tokens=50, temperature=0.8, top_k=30, top_p=0.9,
                                     eos_token_id=gpt2_tok.eos_token_id)
    generated = gpt2_tok.decode(full_ids[0].tolist(), skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Story:  {generated[:300]}")
    print()

    # ── Save ──
    ckpt_dir = Path("checkpoints/tiny_stories_benchmark")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from src.utils.checkpoint import save_checkpoint
    save_checkpoint(ckpt_dir / "final.pt", model, opt, None,
                    step=global_step, epoch=0,
                    config={"benchmark": "TinyStories", "val_ppl": float(final_ppl)})

    import json
    with open(ckpt_dir / "results.json", "w") as f:
        json.dump({
            "model_params": n_params,
            "total_tokens": global_step * bs * seq_len,
            "train_loss_start": losses[0],
            "train_loss_end": losses[-1],
            "val_ppl": float(final_ppl),
            "time_seconds": elapsed,
        }, f, indent=2)

    print(f"✅ Saved: {ckpt_dir / 'results.json'}")


if __name__ == "__main__":
    main()
