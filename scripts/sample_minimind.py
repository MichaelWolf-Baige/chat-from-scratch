#!/usr/bin/env python
"""Sample MiniMind data to match a target token count.

Usage:
    python scripts/sample_minimind.py --target_tokens 15000000 -o data/minimind_sampled.jsonl
"""
import argparse, json, sys
from pathlib import Path
from tokenizers import Tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tokens", type=int, required=True, help="Target total tokens")
    parser.add_argument("--input", default="~/minimind-master/dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("-o", "--output", default="data/minimind_sampled.jsonl")
    parser.add_argument("--tokenizer", default="tokenizers/phase1_8k_real/tokenizer.json")
    args = parser.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output)

    total_tokens = 0
    sampled = 0

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if len(text) < 30:
                    continue
            except:
                continue

            ids = tok.encode(text).ids
            # BOS + tokens + EOS (matching train_single.py convention)
            doc_tokens = 2 + len(ids)  # BOS(1) + tokens + EOS(1)

            if total_tokens + doc_tokens > args.target_tokens:
                # Stop: we'd overshoot
                break

            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total_tokens += doc_tokens
            sampled += 1

    print(f"Sampled {sampled} docs, ~{total_tokens:,} tokens (target: {args.target_tokens:,})")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
