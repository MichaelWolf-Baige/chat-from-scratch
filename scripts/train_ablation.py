#!/usr/bin/env python
"""Ablation experiment runner — train on mixed datasets, evaluate on fixed validation set.

For each mixed dataset (different template/real ratios):
  - Train 14M model for 10 epochs
  - Evaluate on the FIXED validation set (data/val/val_set.jsonl) every 500 steps
  - Report VALIDATION PPL (not training PPL) — the real metric
  - Save best checkpoint based on validation PPL

Usage:
  # Single dataset
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/train_ablation.py \
    --data_file data/mixed/ratio_90_10.jsonl --run_name ablation_90_10

  # All ablation ratios (run sequentially, or submit as separate jobs)
  for ratio in 100_0 90_10 80_20 70_30 50_50; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_ablation.py \
      --data_file data/mixed/ratio_${ratio}.jsonl --run_name ablation_${ratio}
  done
"""

import os, sys, time, json, argparse
from pathlib import Path
from datetime import datetime

import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.utils.checkpoint import save_checkpoint, load_checkpoint

SEED = 42

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_pretrain_texts(data_file: str) -> list[str]:
    """Load texts from a JSONL file for pretraining. Returns plain text strings."""
    texts = []
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"Loading {data_file}...")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                if text and len(text) > 20:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(texts):,} texts, {sum(len(t) for t in texts):,} chars")
    return texts


def load_validation_texts(val_file: str) -> list[str]:
    """Load fixed validation set texts."""
    texts = []
    val_path = Path(val_file)
    if not val_path.exists():
        print(f"WARNING: Validation file {val_file} not found. Skipping val evaluation.")
        return texts

    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                if text and len(text) > 20:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    print(f"  Validation: {len(texts):,} texts loaded")
    return texts


def tokenize_texts(texts: list[str], tokenizer) -> list[int]:
    """Tokenize all texts into a flat list of token IDs with BOS/EOS."""
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        all_ids.append(1)  # BOS
        all_ids.extend(ids)
        all_ids.append(2)  # EOS
    return all_ids


class SeqDataset(torch.utils.data.Dataset):
    """Simple dataset: each item is a fixed-length sequence of token IDs."""
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        inp = self.tokens[idx]
        return {"input_ids": inp, "labels": inp.clone()}


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_on_validation(model, val_loader, device, max_batches: int = 20) -> float:
    """Compute perplexity on the fixed validation set."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            _, out = model(input_ids, labels=labels)
            losses.append(out["loss"].item())
    model.train()
    if not losses:
        return float("inf")
    return float(np.exp(np.mean(losses)))


def train(args):
    """Main training loop with validation set evaluation."""
    # ── DDP setup ──
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ── Reproducibility ──
    s = SEED + rank
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(s)

    if rank == 0:
        print("=" * 60)
        print(f"Ablation Experiment: {args.run_name}")
        print(f"  Data: {args.data_file}")
        print(f"  GPUs: {world}")
        print("=" * 60)

    # ── Load tokenizer ──
    from tokenizers import Tokenizer as HFTokenizer
    tok_path = args.tokenizer_path
    if not os.path.exists(tok_path):
        tok_path = "saved_models/tokenizers/phase1_8k_real_tokenizer.json"
    tok = HFTokenizer.from_file(tok_path)

    # ── Load data ──
    texts = load_pretrain_texts(args.data_file)

    # Tokenize
    if rank == 0:
        print(f"Tokenizing {len(texts):,} texts...")
    token_ids = tokenize_texts(texts, tok)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    if rank == 0:
        unique = len(torch.unique(tokens))
        print(f"  Tokens: {len(tokens):,} total, {unique} unique / {tok.get_vocab_size()} ({100*unique/tok.get_vocab_size():.1f}%)")

    # Chunk into sequences
    seq_len = min(args.seq_len, 1024)
    usable = (len(tokens) // seq_len) * seq_len
    chunks = tokens[:usable].view(-1, seq_len)
    split = int(len(chunks) * 0.95)

    # ── Create dataloaders ──
    train_ds = SeqDataset(chunks[:split], seq_len)
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                       shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.bs, sampler=train_sampler,
        num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # ── Validation loader (from FIXED validation set) ──
    val_texts = load_validation_texts(args.val_file)
    val_loader = None
    if val_texts and rank == 0:
        val_ids = tokenize_texts(val_texts, tok)
        val_tokens = torch.tensor(val_ids, dtype=torch.long)
        val_usable = (len(val_tokens) // seq_len) * seq_len
        val_chunks = val_tokens[:val_usable].view(-1, seq_len)
        val_ds = SeqDataset(val_chunks, seq_len)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.bs, shuffle=False,
            num_workers=1, pin_memory=True)

    # ── Model: 14M ──
    cfg = ModelConfig.phase1()
    cfg.max_seq_len = seq_len
    if args.from_checkpoint:
        model = Transformer(cfg).to(device)
        ckpt = torch.load(args.from_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if rank == 0:
            print(f"  Loaded checkpoint: {args.from_checkpoint}")
    else:
        model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,
                gradient_as_bucket_view=True)
    model.train()

    if rank == 0:
        print(f"  Model: {cfg.total_params:,} params | d={cfg.d_model} L={cfg.n_layers}")

    # ── Optimizer & scheduler ──
    tokens_per_step = args.bs * world * seq_len
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    max_lr = args.lr
    warmup = min(total_steps // 10, 500)
    decay_start = int(total_steps * 0.85)

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95),
                           weight_decay=args.weight_decay)

    if rank == 0:
        print(f"\nTraining: {args.epochs} epochs, ~{total_steps} steps")
        print(f"  Tokens/step: {tokens_per_step:,} | Total: ~{tokens_per_step*total_steps/1e9:.2f}B")
        print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")

    # ── Training loop ──
    gs = 0
    best_val_ppl = float("inf")
    t0 = time.time()
    val_ppls = []  # Track for convergence detection

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if gs >= total_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            _, out = model(input_ids, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            # LR schedule: warmup -> constant -> cosine decay
            if gs < warmup:
                lr = max_lr * (gs + 1) / warmup
            elif gs < decay_start:
                lr = max_lr
            else:
                p = min((gs - decay_start) / max(total_steps - decay_start, 1), 1.0)
                lr = max_lr * 0.01 + 0.5 * max_lr * (1.0 + np.cos(np.pi * p))
            for pg in opt.param_groups:
                pg["lr"] = lr
            gs += 1

            # Logging
            if rank == 0 and (gs <= 20 or gs % 200 == 0):
                elapsed = time.time() - t0
                tps = gs * tokens_per_step / max(elapsed, 0.01)
                train_ppl = np.exp(loss.item())
                print(f"  step {gs:6d}/{total_steps} | loss={loss.item():.4f} "
                      f"train_ppl={train_ppl:.1f} | {tps/1000:.0f}K tok/s "
                      f"| {gs*tokens_per_step/1e9:.2f}B tok")

            # Validation evaluation
            if rank == 0 and val_loader and gs % 500 == 0:
                val_ppl = evaluate_on_validation(model, val_loader, local_rank)
                val_ppls.append((gs, val_ppl))
                print(f"  >>> VAL PPL @ step {gs}: {val_ppl:.1f} "
                      f"(best: {best_val_ppl:.1f}) <<<")

                # Save best
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    save_best(model.module, opt, gs, val_ppl, args, total_tokens=gs * tokens_per_step)

                # Convergence check — only after consuming significant data
                if len(val_ppls) >= 10 and gs > total_steps * 0.3:
                    recent = [p for _, p in val_ppls[-8:]]
                    if all(p >= min(recent) * 0.995 for p in recent):
                        if rank == 0:
                            print("  >>> Validation PPL plateaued after significant training. Early stopping. <<<")
                        break

            # Regular checkpoint
            if rank == 0 and gs % 2000 == 0:
                save_checkpoint(
                    Path(args.output_dir) / f"{args.run_name}_step{gs}.pt",
                    model.module, opt, None, step=gs, epoch=epoch, config={})

        if rank == 0 and gs >= total_steps:
            break

    # ── Final evaluation ──
    if rank == 0:
        elapsed = time.time() - t0
        final_val_ppl = float("inf")
        if val_loader:
            final_val_ppl = evaluate_on_validation(model, val_loader, local_rank, max_batches=50)

        # Save final
        save_checkpoint(
            Path(args.output_dir) / f"{args.run_name}_final.pt",
            model.module, opt, None, step=gs, epoch=0,
            config={"run": args.run_name, "val_ppl": float(final_val_ppl),
                    "tokens": gs * tokens_per_step})

        # Write results summary
        results = {
            "run_name": args.run_name,
            "data_file": args.data_file,
            "model_params": cfg.total_params,
            "total_tokens_trained": gs * tokens_per_step,
            "final_train_ppl": float(np.exp(loss.item())),
            "final_val_ppl": float(final_val_ppl),
            "best_val_ppl": float(best_val_ppl),
            "training_time_hours": round(elapsed / 3600, 2),
            "val_ppls": [(int(s), float(p)) for s, p in val_ppls],
        }
        results_path = Path(args.output_dir) / f"{args.run_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"ABLATION COMPLETE: {args.run_name}")
        print(f"  Time: {elapsed/3600:.1f}hr")
        print(f"  Final VAL PPL: {final_val_ppl:.1f}")
        print(f"  Best VAL PPL:  {best_val_ppl:.1f}")
        print(f"  Results: {results_path}")
        print(f"{'=' * 60}")

    dist.destroy_process_group()


def save_best(model, opt, step, val_ppl, args, total_tokens):
    """Save the best checkpoint based on validation PPL."""
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = Path(args.output_dir) / f"{args.run_name}_best.pt"
    save_checkpoint(path, model, opt, None, step=step, epoch=0,
                    config={"run": args.run_name, "val_ppl": float(val_ppl),
                            "tokens": total_tokens})


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ablation experiment training")
    # Data
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to pretraining JSONL file")
    parser.add_argument("--val_file", type=str, default="data/val/val_set.jsonl",
                       help="Path to fixed validation set")
    parser.add_argument("--tokenizer_path", type=str,
                       default="saved_models/tokenizers/phase1_8k_real_tokenizer.json")
    # Training
    parser.add_argument("--run_name", type=str, required=True,
                       help="Name for this ablation run")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ablation",
                       help="Output directory for checkpoints")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--bs", type=int, default=8,
                       help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--from_checkpoint", type=str, default="",
                       help="Resume from checkpoint")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
