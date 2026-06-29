#!/usr/bin/env python
"""Train 100M model on JSONL data — single GPU, resilient edition.

Changes from original:
  - Intermediate checkpoint every N steps (keeps last K + best val loss)
  - NaN detection — skip bad step (zero_grad + continue)
  - rotate_ckpts sorted by mtime (fixes lexicographic sort bug: 
    step10000 would sort before step8500 in string order, 
    causing the just-saved checkpoint to be immediately deleted)
  - Optimizer state in checkpoint using copy.deepcopy 
    (opt.state_dict() shares tensor refs with opt.state internals)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_single.py -d data/distill_merged.jsonl -o checkpoints/p3_ours.pt -e 2 --save_every 500
"""
import sys, argparse, json, time, os, glob, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, numpy as np, random
from src.model.config import ModelConfig
from src.model.transformer import Transformer

random.seed(42)


def load_texts(path, max_docs=None):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
                if len(t) >= 30:
                    texts.append(t)
            except:
                pass
            if max_docs and len(texts) >= max_docs:
                break
    return texts


def save_ckpt(model, opt, gs, epoch, loss_val, lr_val, data_path, ckpt_path, val_ppl=None):
    """Save checkpoint — no load verification (fast)."""
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()

    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # deepcopy: opt.state_dict() shares tensor refs with opt.state internals
    opt_state = copy.deepcopy(opt.state_dict())
    for pid, pstate in opt_state.get("state", {}).items():
        for k in list(pstate.keys()):
            if isinstance(pstate[k], torch.Tensor):
                pstate[k] = pstate[k].cpu()

    ckpt = {
        "model": state_dict, "optimizer": opt_state,
        "steps": gs, "epoch": epoch,
        "loss": float(loss_val) if loss_val is not None else 0.0,
        "lr": float(lr_val), "data": data_path,
    }
    if val_ppl is not None:
        ckpt["val_ppl"] = float(val_ppl)

    torch.save(ckpt, ckpt_path)
    return str(ckpt_path)


def rotate_ckpts(base_path, keep_count, protect=None):
    """Delete old intermediate checkpoints, keeping last keep_count.
    
    Sorts by modification time (NOT filename) to fix lexicographic bug:
    step10000.pt would sort before step8500.pt in string order, 
    causing the just-saved checkpoint to be immediately deleted.
    
    If protect is given, that file is never removed.
    """
    pattern = str(Path(base_path).parent / f"{Path(base_path).stem}_step*.pt")
    existing = sorted(glob.glob(pattern), key=os.path.getmtime)
    if len(existing) > keep_count:
        for old in existing[:-keep_count]:
            if protect and os.path.abspath(old) == os.path.abspath(protect):
                continue
            try:
                os.remove(old)
            except:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-o", "--output", default="checkpoints/model.pt")
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-b", "--bs", type=int, default=8)
    parser.add_argument("--max_docs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sl", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save intermediate ckpt every N steps (0=only final)")
    parser.add_argument("--keep_ckpts", type=int, default=3,
                        help="Keep last N intermediate checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    texts = load_texts(args.data, args.max_docs)
    random.shuffle(texts)
    all_ids = []
    for t in texts:
        ids = tok.encode(t).ids
        all_ids.append(1)
        all_ids.extend(ids)
        all_ids.append(2)
    tokens = torch.tensor(all_ids, dtype=torch.long)

    print(f"\n{'='*55}\nSINGLE-GPU TRAINING: {args.data}")
    print(f"  {len(texts):,} texts -> {len(tokens):,} tokens")
    if args.save_every > 0:
        print(f"  Save every {args.save_every} steps | keep last {args.keep_ckpts}")
    print(f"{'='*55}")

    # Build dataset
    sl = args.sl
    bs = args.bs
    u = (len(tokens) // sl) * sl
    tf = tokens[:u].view(-1, sl)
    sp = int(len(tf) * 0.95)
    train_t, val_t = tf[:sp], tf[sp:]
    steps_per_epoch = len(train_t) // bs
    total_steps = steps_per_epoch * args.epochs

    # Model
    cfg = ModelConfig(
        vocab_size=8192, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, d_ff=args.d_ff,
        max_seq_len=1024, rope_theta=100000.0, dropout=0.0,
        use_flash_attention=True, tie_word_embeddings=True,
        rms_norm_eps=1e-6, use_qk_norm=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = Transformer(cfg).to(device)
    model.train()
    n = cfg.total_params
    print(f"  {n:,} params | {total_steps} steps | bs={bs}x{sl} | {args.epochs} epochs | LR={args.lr}")

    # Optimizer
    max_lr = args.lr
    warmup = total_steps // 10
    decay_start = int(total_steps * 0.85)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    gs = 0
    t0 = time.time()
    nan_count = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_t))
        for i in range(0, len(train_t) - bs, bs):
            if gs >= total_steps:
                break

            idx = perm[i:i + bs]
            batch = train_t[idx].to(device)
            _, out = model(batch, labels=batch)
            loss = out["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  *** NaN/Inf at step {gs} (event #{nan_count}) ***")
                opt.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

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

            if gs <= 10 or gs % 50 == 0:
                el = time.time() - t0
                lc = min(loss.item(), 10)
                print(f"  step {gs:5d}/{total_steps} | loss={loss.item():.4f} ppl={np.exp(lc):.0f} | {gs * bs * sl / max(el, 0.01) / 1000:.0f}K tok/s")

            if args.save_every > 0 and gs % args.save_every == 0:
                ckpt_name = f"{Path(args.output).stem}_step{gs}.pt"
                ckpt_path = str(Path(args.output).parent / ckpt_name)
                saved = save_ckpt(model, opt, gs, epoch, loss.item(), lr, args.data, ckpt_path)
                if saved:
                    rotate_ckpts(args.output, args.keep_ckpts, protect=saved)
                    print(f"  [ckpt] {ckpt_name}")

                model.eval()
                with torch.no_grad():
                    n_v = min(bs, len(val_t))
                    vb = val_t[:n_v].to(device)
                    _, veo = model(vb, labels=vb)
                    vloss = veo["loss"].item()
                model.train()
                if vloss < best_val_loss:
                    best_val_loss = vloss
                    best_name = f"{Path(args.output).stem}_best.pt"
                    best_path = str(Path(args.output).parent / best_name)
                    save_ckpt(model, opt, gs, epoch, loss.item(), lr, args.data, best_path, val_ppl=vloss)
                    print(f"  [best] {best_name} (val_loss={vloss:.4f})")

    model.eval()
    et = []
    with torch.no_grad():
        for i in range(0, min(len(val_t), bs * 20), bs):
            vb = val_t[i:i + bs].to(device)
            _, eo = model(vb, labels=vb)
            et.append(eo["loss"].item())
    val_ppl = np.exp(np.mean(et))
    elapsed = time.time() - t0

    saved = save_ckpt(model, opt, gs, args.epochs, None, lr, args.data, args.output, val_ppl=val_ppl)

    print(f"\n{'='*55}\nDONE: {args.output}")
    print(f"  VAL PPL: {val_ppl:.1f} | {elapsed/60:.1f}min | NaN events: {nan_count}")
    if saved:
        print(f"  Final ckpt: {saved} ({Path(saved).stat().st_size/1e9:.2f}GB)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()