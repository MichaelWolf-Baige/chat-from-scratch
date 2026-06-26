#!/usr/bin/env python
"""Quick chat test with SFT checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

tok = Tokenizer.from_file("tokenizers/phase1_8k_real/tokenizer.json")
cfg = ModelConfig(vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
                   d_ff=1024, max_seq_len=512, dropout=0.0, use_flash_attention=True,
                   tie_word_embeddings=True, rms_norm_eps=1e-6,
                   pad_token_id=0, bos_token_id=1, eos_token_id=2)
model = Transformer(cfg).cuda()
ckpt = torch.load("checkpoints/sft_chat/final.pt", map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
nparam = sum(pp.numel() for pp in model.parameters())
print(f"Loaded: SFT model, {nparam:,} params")

prompts = [
    "你好！请介绍一下什么是算法。",
    "我想学习编程，应该从哪里开始？",
    "谢谢你今天的帮助！",
]

for prompt in prompts:
    text = f"用户：{prompt}\n助手："
    ids = [1] + tok.encode(text).ids
    pid = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        full, _ = model.generate(pid, max_new_tokens=60, temperature=0.8, top_k=30, top_p=0.9, eos_token_id=2)
    result = tok.decode(full[0].tolist(), skip_special_tokens=True)
    parts = result.split("助手：")
    resp = parts[-1].strip() if len(parts) > 1 else result
    print(f"Q: {prompt}")
    print(f"A: {resp[:200]}")
    print()
