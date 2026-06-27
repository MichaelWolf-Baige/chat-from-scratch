#!/usr/bin/env python
"""Quick eval: train 100M model on our distilled data, test generation.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/eval_our_data.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np, random, time, json
from src.model.config import ModelConfig
from src.model.transformer import Transformer
random.seed(42)

def main():
    dist.init_process_group(backend="nccl"); rank=dist.get_rank(); world=dist.get_world_size()
    local_r=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_r); device=torch.device(f"cuda:{local_r}")
    torch.manual_seed(42+rank); torch.cuda.manual_seed_all(42+rank)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    random.seed(42)

    from tokenizers import Tokenizer as HFTok; tok=HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    # Load data
    texts=[]; n=min(50000, 87395)
    with open("data/distill_merged.jsonl",encoding="utf-8") as f:
        for line in f:
            try: t=json.loads(line)["text"]; texts.append(t) if len(t)>=30 else None
            except: pass
            if len(texts)>=n: break
    random.shuffle(texts)

    all_ids=[]
    for t in texts:
        ids=tok.encode(t).ids; all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    tokens=torch.tensor(all_ids,dtype=torch.long)

    if rank==0: print(f"\n{'='*55}\nOUR DISTILLED DATA: {len(texts):,} texts, {len(tokens):,} tokens\n{'='*55}")

    # Model
    cfg=ModelConfig(vocab_size=8192,d_model=512,n_layers=24,n_heads=8,n_kv_heads=4,d_ff=2048,max_seq_len=1024,rope_theta=100000.0,dropout=0.0,use_flash_attention=True,tie_word_embeddings=True,rms_norm_eps=1e-6,use_qk_norm=True,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    model=Transformer(cfg).to(device)
    model=DDP(model,device_ids=[local_r],find_unused_parameters=False,gradient_as_bucket_view=True)
    model.train()

    # Train/val
    sl=1024; bs=12; u=(len(tokens)//sl)*sl
    tf=tokens[:u].view(-1,sl); sp=int(len(tf)*0.95)
    class DS(torch.utils.data.Dataset):
        def __init__(self,tk,sl): self.t=tk; self.s=sl
        def __len__(self): return len(self.t)
        def __getitem__(self,i): return {"input_ids":self.t[i],"labels":self.t[i].clone()}
    tr=DS(tf[:sp],sl); vl=DS(tf[sp:],sl)
    ts=DistributedSampler(tr,num_replicas=world,rank=rank,shuffle=True,drop_last=True)
    vs=DistributedSampler(vl,num_replicas=world,rank=rank,shuffle=False,drop_last=True)
    tl=torch.utils.data.DataLoader(tr,batch_size=bs,sampler=ts,num_workers=2,pin_memory=True,prefetch_factor=2,persistent_workers=True)
    vl=torch.utils.data.DataLoader(vl,batch_size=bs,sampler=vs,num_workers=2,pin_memory=True,prefetch_factor=2,persistent_workers=True)

    tps=bs*world*sl; ts_steps=len(tl)*2; mlr=5e-4; wu=ts_steps//10; dc=int(ts_steps*0.85)
    opt=torch.optim.AdamW(model.parameters(),lr=mlr,betas=(0.9,0.95),weight_decay=0.1)
    gs=0; t0=time.time()

    if rank==0: print(f"Training: {ts_steps} steps, ~{ts_steps*tps/1e9:.2f}B tokens, LR={mlr} WSD")

    for epoch in range(2):
        ts.set_epoch(epoch)
        for batch in tl:
            if gs>=ts_steps: break
            iid=batch["input_ids"].to(device,non_blocking=True); lbl=batch["labels"].to(device,non_blocking=True)
            _,out=model(iid,labels=lbl); loss=out["loss"]; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); opt.zero_grad()
            if gs<wu: lr=mlr*(gs+1)/wu
            elif gs<dc: lr=mlr
            else: p=min((gs-dc)/max(ts_steps-dc,1),1.0); lr=mlr*0.01+0.5*mlr*(1.0+np.cos(np.pi*p))
            for pg in opt.param_groups: pg["lr"]=lr
            losses_val=loss.item(); gs+=1
            if rank==0 and gs%50==0:
                el=time.time()-t0; print(f"  step {gs:5d}/{ts_steps} | loss={losses_val:.4f} ppl={np.exp(losses_val):.0f} | {gs*tps/el/1000:.0f}K tok/s")

    # Save checkpoint FIRST (before any GPU-heavy ops that might OOM)
    dist.barrier()  # sync all ranks before checkpoint
    if rank==0:
        ckpt_path=Path("checkpoints/eval_our_data/final.pt"); ckpt_path.parent.mkdir(parents=True,exist_ok=True)
        model_state = model.module.state_dict() if hasattr(model,"module") else model.state_dict()
        torch.save({"model":model_state}, ckpt_path)
        print(f"💾 Saved: {ckpt_path}")

    # Final eval (rank 0 only)
    if rank==0:
        model.eval(); et=[]
        with torch.no_grad():
            for ei,eb in enumerate(vl):
                if ei>=15: break
                _,eo=model(eb["input_ids"].to(device),labels=eb["labels"].to(device)); et.append(eo["loss"].item())
        vppl=np.exp(np.mean(et)); elapsed=time.time()-t0
        print(f"\n{'='*55}\nVAL PPL: {vppl:.0f} | {elapsed/60:.1f}min | {gs*tps/el/1000:.0f}K tok/s")

        # Gen test (single GPU — no OOM)
        print(f"\nGenerative Test:")
        for prompt in ["人工智能是","北京是中国的","春天来了，","什么是机器学习？","今天天气"]:
            ids=tok.encode(prompt).ids; pid=torch.tensor([[1]+ids],device=device)
            out_tokens=[]
            for tid,isdone in model.module.generate_stream(pid,max_new_tokens=30,temperature=0.8,top_k=35,top_p=0.9,eos_token_id=2):
                out_tokens.append(tid)
                if isdone: break
            print(f"  {prompt} {tok.decode(out_tokens,skip_special_tokens=True)[:80]}")
        print(f"\n✅ Done!")

    dist.destroy_process_group()

if __name__=="__main__": main()
