#!/usr/bin/env python
"""Generate text from a trained model.

Usage:
    python scripts/generate.py \
        --checkpoint checkpoints/phase1/step_10000.pt \
        --prompt "你好，我是" \
        --max_new_tokens 100 \
        --temperature 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.tokenizer_utils import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Training config (auto-detect from checkpoint dir if not given)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tokenizers/phase1_bpe_8k/tokenizer.json",
        help="Path to tokenizer.json"
    )
    parser.add_argument(
        "--prompt", type=str, default="你好", help="Input prompt"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (lower = more deterministic)"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-K sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )

    args = parser.parse_args()

    # Load tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Train one first: python scripts/train_tokenizer.py --data_dir data/raw/")
        return

    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")

    # Infer model config from training config
    ckpt_path = Path(args.checkpoint)
    if args.config is None:
        # Try to find config in checkpoint directory's parent
        config_candidates = [
            ckpt_path.parent / ".." / ".." / "configs" / "train" / "phase1.yaml",
            Path("configs/train/phase1.yaml"),
        ]
        args.config = None
        for c in config_candidates:
            resolved = c.resolve()
            if resolved.exists():
                args.config = str(resolved)
                break
        if args.config is None:
            print("Could not auto-detect config. Use --config to specify.")
            return

    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)

    model_cfg_path = train_cfg.get("model_config", "configs/model/phase1.yaml")
    model_cfg_full = Path(args.config).parent.parent / model_cfg_path
    if model_cfg_full.exists():
        with open(model_cfg_full, "r") as f:
            model_cfg = yaml.safe_load(f)
    else:
        # Use defaults
        model_cfg = {"vocab_size": 8192, "d_model": 384, "n_layers": 6,
                     "n_heads": 6, "n_kv_heads": 6, "d_ff": 1024,
                     "max_seq_len": 2048}

    # Build model
    model_config = ModelConfig(**{
        k: v for k, v in model_cfg.items()
        if k in ModelConfig.__dataclass_fields__
    })
    model = Transformer(model_config)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Model loaded from step {checkpoint.get('step', '?')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Encode prompt
    prompt = args.prompt
    input_ids = torch.tensor(
        [tokenizer.encode(prompt).ids], dtype=torch.long, device=device
    )
    print(f"\nPrompt: '{prompt}' ({input_ids.shape[1]} tokens)")
    print("=" * 60)

    # Generate
    for i in range(args.num_samples):
        with torch.no_grad():
            full_ids, new_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=tokenizer.token_to_id("</s>") or 2,
            )

        generated_text = tokenizer.decode(new_ids[0].tolist(), skip_special_tokens=True)
        full_text = tokenizer.decode(full_ids[0].tolist(), skip_special_tokens=True)

        if args.num_samples > 1:
            print(f"\n--- Sample {i+1} ---")
        print(f"Generated: {generated_text}")
        print(f"\nFull text:\n{full_text}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
