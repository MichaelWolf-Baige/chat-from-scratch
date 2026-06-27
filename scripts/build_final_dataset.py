#!/usr/bin/env python
"""Build the final pretraining dataset:
  1. Load real data (wiki + news + dialogs)
  2. MD5 exact dedup
  3. MinHash near-dedup (threshold 0.80)
  4. Mix in ~5% dialog-format data (from SFT conversations)
  5. Write final JSONL

Usage: python scripts/build_final_dataset.py
"""

import json, hashlib, random, re, sys
from pathlib import Path
from collections import Counter

random.seed(42)
BASE = Path(__file__).parent.parent

def main():
    real_dir = BASE / "data" / "real"
    sft_file = BASE / "data" / "sft" / "sft_dialogs.jsonl"
    out_dir = BASE / "data" / "final"
    out_dir.mkdir(exist_ok=True)

    # ═══ 1. Load all real data ═══
    print("[1/5] Loading real data...")
    real_texts = []
    for f in sorted(real_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fp:
            for line in fp:
                d = json.loads(line.strip())
                t = d.get("text", "").strip()
                if t and len(t) > 30:
                    real_texts.append(t)
    print("  Loaded: {} texts".format(len(real_texts)))

    # ═══ 2. MD5 exact dedup ═══
    print("[2/5] MD5 exact dedup...")
    seen = set()
    deduped = []
    for t in real_texts:
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(t)
    print("  {} -> {} ({} removed)".format(len(real_texts), len(deduped), len(real_texts)-len(deduped)))

    # ═══ 3. MinHash near-dedup ═══
    print("[3/5] MinHash near-dedup (threshold=0.80)...")

    def char_ngrams(text, n=5):
        return set(text[i:i+n] for i in range(len(text)-n+1))

    def jaccard(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    ngram_sets = [char_ngrams(t) for t in deduped]
    if len(deduped) > 1:
        num_hashes = 64
        max_hash = 2**31 - 1
        seeds = [random.randint(1, max_hash) for _ in range(num_hashes)]

        signatures = []
        for ngs in ngram_sets:
            sig = []
            for seed in seeds:
                if ngs:
                    sig.append(min((hash(ng) ^ seed) % max_hash for ng in ngs))
                else:
                    sig.append(max_hash)
            signatures.append(sig)

        # LSH banding
        band_size = 4
        num_bands = num_hashes // band_size
        candidates = set()

        for band in range(num_bands):
            bucket = {}
            start = band * band_size
            sl = slice(start, start + band_size)
            for idx, sig in enumerate(signatures):
                key = tuple(sig[sl])
                if key in bucket:
                    for other in bucket[key]:
                        candidates.add((min(idx, other), max(idx, other)))
                    bucket[key].append(idx)
                else:
                    bucket[key] = [idx]

        to_remove = set()
        for i, j in sorted(candidates):
            if i in to_remove or j in to_remove:
                continue
            sim = jaccard(ngram_sets[i], ngram_sets[j])
            if sim >= 0.80:
                to_remove.add(j)

        near_deduped = [t for idx, t in enumerate(deduped) if idx not in to_remove]
    else:
        near_deduped = deduped

    print("  {} -> {} ({} removed)".format(len(deduped), len(near_deduped), len(deduped)-len(near_deduped)))

    # ═══ 4. Add dialog-format data (~5%) ═══
    print("[4/5] Adding dialog-format data...")
    dialog_texts = []
    if sft_file.exists():
        with open(sft_file, encoding="utf-8") as fp:
            for line in fp:
                d = json.loads(line.strip())
                conv = d.get("conversations", [])
                parts = []
                for turn in conv:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role == "user":
                        parts.append("用户：{}".format(content))
                    elif role == "assistant":
                        parts.append("助手：{}".format(content))
                if parts:
                    dialog_texts.append("\n".join(parts))
    print("  Dialog texts available: {}".format(len(dialog_texts)))

    n_dialog = max(int(len(near_deduped) * 0.05), min(len(dialog_texts), 2000))
    if n_dialog > 0:
        if n_dialog <= len(dialog_texts):
            dialog_sample = random.sample(dialog_texts, n_dialog)
        else:
            dialog_sample = random.choices(dialog_texts, k=n_dialog)
    else:
        dialog_sample = []

    all_texts = near_deduped + dialog_sample
    random.shuffle(all_texts)
    print("  Final: {} texts ({} real + {} dialog)".format(len(all_texts), len(near_deduped), len(dialog_sample)))

    # ═══ 5. Write ═══
    print("[5/5] Writing final dataset...")
    out_file = out_dir / "pretrain_final.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    # Stats
    lengths = [len(t) for t in all_texts]
    total_chars = sum(lengths)
    cjk_counts = [sum(1 for c in t if '一' <= c <= '鿿') for t in all_texts]
    avg_cjk = sum(c / max(len(t), 1) for t, c in zip(all_texts, cjk_counts)) / len(all_texts)

    sep = "=" * 50
    print("")
    print(sep)
    print("FINAL DATASET")
    print(sep)
    print("  Total: {:,} texts".format(len(all_texts)))
    print("  Size:  {:.1f} MB".format(total_chars / 1024 / 1024))
    print("  Length: mean={:.0f} median={} [{}-{}]".format(
        sum(lengths)/len(lengths), sorted(lengths)[len(lengths)//2], min(lengths), max(lengths)))
    print("  Avg CJK ratio: {:.1%}".format(avg_cjk))
    print("  Composition: {:,} real + {:,} dialog ({:.0%} dialog)".format(
        len(near_deduped), len(dialog_sample), len(dialog_sample)/len(all_texts)))
    print("  Output: {}".format(out_file.resolve()))

if __name__ == "__main__":
    main()
