#!/usr/bin/env python
"""Chat with the trained SFT model — streaming token-by-token output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

TOKENIZER_PATH = "saved_models/tokenizers/phase1_8k_real_tokenizer.json"
CHECKPOINT_PATH = "saved_models/sft_chat_final.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tok = Tokenizer.from_file(TOKENIZER_PATH)
cfg = ModelConfig(vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
                   d_ff=1024, max_seq_len=512, dropout=0.0,
                   use_flash_attention=(device.type == "cuda"),
                   tie_word_embeddings=True, rms_norm_eps=1e-6,
                   pad_token_id=0, bos_token_id=1, eos_token_id=2)
model = Transformer(cfg)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model = model.to(device)
model.eval()
nparam = sum(pp.numel() for pp in model.parameters())
print(f"Loaded: SFT chat model, {nparam:,} params")
print("=" * 40)
print("Chat from Scratch - 14M Chinese Chat")
print("Type /quit to exit")
print("=" * 40)

while True:
    try:
        user_input = input("\nYou: ")
    except (EOFError, KeyboardInterrupt):
        break
    if user_input.lower() in ("/quit", "/exit", "/q"):
        break
    if not user_input.strip():
        continue

    text = f"用户：{user_input}\n助手："
    ids = [1] + tok.encode(text).ids
    pid = torch.tensor([ids], device=device)

    # Stream token by token
    sys.stdout.write("Bot: ")
    sys.stdout.flush()
    generated_text = ""
    with torch.no_grad():
        for token_id, is_done in model.generate_stream(
            pid, max_new_tokens=80, temperature=0.8, top_k=35, top_p=0.9, eos_token_id=2
        ):
            generated_text += tok.decode([token_id], skip_special_tokens=True)
            sys.stdout.write(tok.decode([token_id], skip_special_tokens=True))
            sys.stdout.flush()
            if is_done:
                break
    print()  # newline after response
