#!/usr/bin/env python
"""Chat with trained model (ChatML format).

Auto-detects: checkpoint path, CUDA/CPU, ChatML vs natural format.

Usage: python scripts/chat_test.py [--checkpoint saved_models/sft_v5_final.pt]
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

# ── Config ────────────────────────────────────────────────────────
# Auto-detect best available checkpoint
CANDIDATES = [
    "saved_models/sft_v5_final.pt",
    "saved_models/sft_v2_final.pt",
    "saved_models/sft_chat_final.pt",
]
TOKENIZER_PATH = "saved_models/tokenizers/phase1_8k_real_tokenizer.json"

# ChatML special token IDs
BOS_ID = 1
EOS_ID = 2
USER_ID = 5
ASST_ID = 6

# ── Load ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Pick checkpoint
ckpt_path = None
for c in CANDIDATES:
    if Path(c).exists():
        ckpt_path = c
        break
if ckpt_path is None:
    # Also check server path
    server = Path("tokenizers/phase1_8k_real/tokenizer.json")
    if server.exists():
        TOKENIZER_PATH = str(server)
        ckpt_path = "checkpoints/sft_v5/final.pt"

if ckpt_path is None or not Path(ckpt_path).exists():
    print("ERROR: No checkpoint found. Checked:")
    for c in CANDIDATES + ["checkpoints/sft_v5/final.pt"]:
        print(f"  {c} — {'FOUND' if Path(c).exists() else 'MISSING'}")
    sys.exit(1)

print(f"Checkpoint: {ckpt_path}")
tok = Tokenizer.from_file(TOKENIZER_PATH)

# ── Model ──────────────────────────────────────────────────────────
cfg = ModelConfig(
    vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
    d_ff=1024, max_seq_len=512, dropout=0.0,
    use_flash_attention=(device.type == "cuda"),
    tie_word_embeddings=True, rms_norm_eps=1e-6,
    pad_token_id=0, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
)
model = Transformer(cfg)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model = model.to(device)
model.eval()
nparam = sum(pp.numel() for pp in model.parameters())
print(f"Model: {nparam:,} params")

# ── Chat loop ──────────────────────────────────────────────────────
print("=" * 40)
print("Chat from Scratch — 14M Chinese Chat (ChatML)")
print("Type /quit to exit")
print("=" * 40)

while True:
    try:
        user_input = input("\n[You]: ")
    except (EOFError, KeyboardInterrupt):
        break
    if user_input.lower() in ("/quit", "/exit", "/q"):
        break
    if not user_input.strip():
        continue

    # Build ChatML prompt: <s><|user|>text<|assistant|>
    user_ids = tok.encode(user_input).ids
    prompt_ids = [BOS_ID, USER_ID] + user_ids + [ASST_ID]
    pid = torch.tensor([prompt_ids], device=device)

    # Stream response
    sys.stdout.write("[Bot]: ")
    sys.stdout.flush()
    with torch.no_grad():
        for token_id, is_done in model.generate_stream(
            pid, max_new_tokens=80, temperature=0.8,
            top_k=35, top_p=0.9, eos_token_id=EOS_ID
        ):
            text = tok.decode([token_id], skip_special_tokens=True)
            sys.stdout.write(text)
            sys.stdout.flush()
            if is_done:
                break
    print()
