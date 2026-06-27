#!/usr/bin/env python
"""Minimal smoke test: 500 texts → train → save → load → generate."""
import sys; sys.path.insert(0,'.')
import torch, json, numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

texts = []
with open("data/distill_merged.jsonl", encoding="utf-8") as f:
    for line in f:
        try: t=json.loads(line)["text"]; texts.append(t) if len(t)>=30 else None
        except: pass
        if len(texts)>=500: break

tok = Tokenizer.from_file("tokenizers/phase1_8k_real/tokenizer.json")
all_ids=[]
for t in texts: ids=tok.encode(t).ids; all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
tokens=torch.tensor(all_ids,dtype=torch.long)
print(f"Data: {len(texts)} texts, {len(tokens)} tokens")

# Tiny model, 50 steps
cfg=ModelConfig(vocab_size=8192,d_model=128,n_layers=6,n_heads=4,n_kv_heads=4,d_ff=384,max_seq_len=256,
                dropout=0.0,use_flash_attention=True,tie_word_embeddings=True,rms_norm_eps=1e-6,
                pad_token_id=0,bos_token_id=1,eos_token_id=2)
model=Transformer(cfg).cuda(); model.train()
print(f"Model: {cfg.total_params:,} params")

bs=8; sl=256; u=(len(tokens)//sl)*sl; tf=tokens[:u].view(-1,sl)
opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
losses=[]
for step in range(50):
    idx=torch.randint(0,len(tf)-bs,(bs,)); batch=tf[idx].cuda()
    _,out=model(batch,labels=batch); loss=out["loss"]; loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
    losses.append(loss.item())
    if step<=5 or step%10==0: print(f"  step {step:2d}: loss={loss.item():.4f} ppl={np.exp(loss.item()):.0f}")
print(f"Loss: {losses[0]:.2f}->{losses[-1]:.2f} delta={losses[0]-losses[-1]:.1f}")

# Save
p=Path("checkpoints/smoke_test.pt"); p.parent.mkdir(parents=True,exist_ok=True)
torch.save({"model":model.state_dict()},p)
print(f"SAVED: {p} ({p.stat().st_size/1e6:.1f}MB)")

# Load + generate
model2=Transformer(cfg).cuda()
model2.load_state_dict(torch.load(p,map_location="cuda",weights_only=False)["model"]); model2.eval()
for prompt in ["人工智能是","北京是中国的","春天来了"]:
    ids=tok.encode(prompt).ids; pid=torch.tensor([[1]+ids],device="cuda")
    out_tokens=[]
    for tid,is_done in model2.generate_stream(pid,max_new_tokens=20,temperature=0.8,top_k=30,top_p=0.9,eos_token_id=2):
        out_tokens.append(tid)
        if is_done: break
    print(f"  {prompt} -> {tok.decode(out_tokens,skip_special_tokens=True)[:60]}")
print("PIPELINE OK: train->save->load->generate, NO CRASH")
