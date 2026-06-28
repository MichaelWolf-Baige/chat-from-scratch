#!/usr/bin/env python
"""Train 100M model on JSONL data — single GPU, no DDP, rock solid.

Step 1: Train → checkpoint
Step 2: (separate script) Load → generate

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_single.py -d data/distill_merged.jsonl -o checkpoints/p3_ours.pt -e 2
"""
import sys, argparse, json, time
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
                t = json.loads(line).get("text", ""); texts.append(t) if len(t) >= 30 else None
            except: pass
            if max_docs and len(texts) >= max_docs: break
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data",required=True); parser.add_argument("-o","--output",default="checkpoints/model.pt")
    parser.add_argument("-e","--epochs",type=int,default=2); parser.add_argument("-b","--bs",type=int,default=8)
    parser.add_argument("--max_docs",type=int,default=50000); parser.add_argument("--lr",type=float,default=5e-4)
    parser.add_argument("--sl",type=int,default=1024)
    parser.add_argument("--d_model",type=int,default=512); parser.add_argument("--n_layers",type=int,default=24)
    parser.add_argument("--n_heads",type=int,default=8); parser.add_argument("--n_kv_heads",type=int,default=4)
    parser.add_argument("--d_ff",type=int,default=2048)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    from tokenizers import Tokenizer as HFTok; tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    texts = load_texts(args.data, args.max_docs); random.shuffle(texts)
    all_ids = []
    for t in texts: ids=tok.encode(t).ids; all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    tokens = torch.tensor(all_ids, dtype=torch.long)

    print(f"\n{'='*55}\nSINGLE-GPU TRAINING: {args.data}")
    print(f"  {len(texts):,} texts → {len(tokens):,} tokens\n{'='*55}")

    # Build dataset
    sl = args.sl; bs = args.bs
    u = (len(tokens)//sl)*sl; tf = tokens[:u].view(-1,sl); sp = int(len(tf)*0.95)
    train_t, val_t = tf[:sp], tf[sp:]
    steps_per_epoch = len(train_t)//bs; total_steps = steps_per_epoch * args.epochs

    # Model
    cfg = ModelConfig(vocab_size=8192,d_model=args.d_model,n_layers=args.n_layers,n_heads=args.n_heads,n_kv_heads=args.n_kv_heads,d_ff=args.d_ff,max_seq_len=1024,rope_theta=100000.0,dropout=0.0,use_flash_attention=True,tie_word_embeddings=True,rms_norm_eps=1e-6,use_qk_norm=True,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    model = Transformer(cfg).to(device); model.train()
    n = cfg.total_params
    print(f"  {n:,} params | {total_steps} steps | bs={bs}x{sl} | {args.epochs} epochs | LR={args.lr}")

    # Optimizer
    max_lr = args.lr; warmup = total_steps//10; decay_start = int(total_steps*0.85)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9,0.95), weight_decay=0.1)
    gs = 0; t0 = time.time()

    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_t))
        for i in range(0, len(train_t)-bs, bs):
            if gs >= total_steps: break
            idx = perm[i:i+bs]; batch = train_t[idx].to(device)
            _, out = model(batch, labels=batch); loss = out["loss"]; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()

            if gs < warmup: lr = max_lr*(gs+1)/warmup
            elif gs < decay_start: lr = max_lr
            else: p = min((gs-decay_start)/max(total_steps-decay_start,1),1.0); lr = max_lr*0.01+0.5*max_lr*(1.0+np.cos(np.pi*p))
            for pg in opt.param_groups: pg["lr"] = lr
            gs += 1

            if gs <= 10 or gs % 50 == 0:
                el = time.time()-t0
                print(f"  step {gs:5d}/{total_steps} | loss={loss.item():.4f} ppl={np.exp(loss.item()):.0f} | {gs*bs*sl/max(el,0.01)/1000:.0f}K tok/s")

    # ── Final eval ──
    model.eval(); et = []
    with torch.no_grad():
        for i in range(0, min(len(val_t), bs*20), bs):
            vb = val_t[i:i+bs].to(device); _, eo = model(vb, labels=vb); et.append(eo["loss"].item())
    val_ppl = np.exp(np.mean(et)); elapsed = time.time()-t0

    # ── Save ── (move to CPU first to avoid OOM during serialization)
    ckpt_path = Path(args.output); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"model": state_dict, "val_ppl": float(val_ppl), "steps": gs, "data": args.data}, ckpt_path)

    print(f"\n{'='*55}\nDONE: {args.output}")
    print(f"  VAL PPL: {val_ppl:.0f} | {elapsed/60:.1f}min | {gs*bs*sl/elapsed/1000:.0f}K tok/s")
    print(f"  Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size/1e9:.1f}GB)")
    print(f"{'='*55}")

if __name__=="__main__": main()
