#!/usr/bin/env python
"""Step 2: Load trained checkpoint → run generative test on single GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/gen_test.py --checkpoint checkpoints/p3_ours/final.pt
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint .pt file")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Checkpoint: {args.checkpoint}")

    tok = Tokenizer.from_file("tokenizers/phase1_8k_real/tokenizer.json")
    cfg = ModelConfig(vocab_size=8192, d_model=512, n_layers=24, n_heads=8, n_kv_heads=4,
                      d_ff=2048, max_seq_len=1024, rope_theta=100000.0, dropout=0.0,
                      use_flash_attention=(device.type=="cuda"), tie_word_embeddings=True,
                      rms_norm_eps=1e-6, use_qk_norm=True,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = Transformer(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    vppl = ckpt.get("val_ppl", "N/A")
    print(f"Model: {n:,} params | VAL PPL: {vppl}")

    prompts = [
        "人工智能是","北京是中国的","春天来了，","什么是机器学习？",
        "今天天气","中国最大的城市是","请写一首关于秋天的诗","1+1等于几",
    ]
    print(f"\n{'='*55}")
    for prompt in prompts:
        ids = tok.encode(prompt).ids
        pid = torch.tensor([[1]+ids], device=device)
        out_tokens = []
        for tid, is_done in model.generate_stream(
            pid, max_new_tokens=40, temperature=0.8, top_k=35, top_p=0.9, eos_token_id=2
        ):
            out_tokens.append(tid)
            if is_done: break
        resp = tok.decode(out_tokens, skip_special_tokens=True)
        print(f"  {prompt}")
        print(f"  -> {resp[:100]}")
        print()

if __name__ == "__main__":
    main()
