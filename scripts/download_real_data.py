#!/usr/bin/env python
"""Download real pretraining data from HuggingFace datasets.

Sources:
    - FineWeb-Edu (English, quality-filtered web text)
    - Chinese Wikipedia
    - mc4-zh (Chinese web text from C4)

Output: data/raw/*.jsonl — one JSON object per line: {"text": "...", "source": "..."}

Usage:
    python scripts/download_real_data.py --output data/raw/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_fineweb_edu(output_dir: Path, target_mb: int = 250, seed: int = 42):
    """Download FineWeb-Edu English samples via streaming."""
    from datasets import load_dataset

    print(f"\n{'='*50}")
    print(f"📥 FineWeb-Edu (target: ~{target_mb}MB)")
    print(f"{'='*50}")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    # Shuffle with seed, take a fixed number of samples
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    output_path = output_dir / "fineweb_edu_en.jsonl"
    total_chars = 0
    count = 0
    target_chars = target_mb * 1_000_000  # ~250MB chars

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in ds:
            text = sample.get("text", "")
            if not text or len(text) < 200:  # skip very short
                continue
            f.write(json.dumps({"text": text, "source": "fineweb-edu"}, ensure_ascii=False) + "\n")
            total_chars += len(text)
            count += 1

            if count % 500 == 0:
                mb_sofar = total_chars / 1_000_000
                print(f"  {count:6d} docs | {mb_sofar:7.1f} MB | ~{total_chars / max(count, 1):.0f} chars/doc")

            if total_chars >= target_chars:
                break

    file_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Done: {count} docs, {file_mb:.1f} MB, {total_chars:,} chars")
    return {"file": str(output_path), "count": count, "size_mb": file_mb, "chars": total_chars}


def download_chinese_wikipedia(output_dir: Path, target_mb: int = 180, seed: int = 42):
    """Download Chinese Wikipedia via datasets."""
    from datasets import load_dataset

    print(f"\n{'='*50}")
    print(f"📥 Chinese Wikipedia (target: ~{target_mb}MB)")
    print(f"{'='*50}")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.zh",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    output_path = output_dir / "wiki_zh.jsonl"
    total_chars = 0
    count = 0
    target_chars = target_mb * 1_000_000

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in ds:
            text = sample.get("text", "")
            if not text or len(text) < 100:
                continue
            # Strip excessive newlines
            text = text.replace("\n\n\n", "\n\n").strip()
            f.write(json.dumps({"text": text, "source": "wiki-zh"}, ensure_ascii=False) + "\n")
            total_chars += len(text)
            count += 1

            if count % 200 == 0:
                mb_sofar = total_chars / 1_000_000
                print(f"  {count:5d} docs | {mb_sofar:7.1f} MB")

            if total_chars >= target_chars:
                break

    file_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Done: {count} docs, {file_mb:.1f} MB, {total_chars:,} chars")
    return {"file": str(output_path), "count": count, "size_mb": file_mb, "chars": total_chars}


def download_mc4_zh(output_dir: Path, target_mb: int = 70, seed: int = 42):
    """Download mc4-zh samples (Chinese C4 subset)."""
    from datasets import load_dataset

    print(f"\n{'='*50}")
    print(f"📥 mc4-zh (target: ~{target_mb}MB)")
    print(f"{'='*50}")

    ds = load_dataset(
        "allenai/c4",
        "zh",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    output_path = output_dir / "mc4_zh.jsonl"
    total_chars = 0
    count = 0
    target_chars = target_mb * 1_000_000

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in ds:
            text = sample.get("text", "")
            if not text or len(text) < 200:
                continue
            text = text.strip()
            f.write(json.dumps({"text": text, "source": "mc4-zh"}, ensure_ascii=False) + "\n")
            total_chars += len(text)
            count += 1

            if count % 200 == 0:
                mb_sofar = total_chars / 1_000_000
                print(f"  {count:5d} docs | {mb_sofar:7.1f} MB")

            if total_chars >= target_chars:
                break

    file_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Done: {count} docs, {file_mb:.1f} MB, {total_chars:,} chars")
    return {"file": str(output_path), "count": count, "size_mb": file_mb, "chars": total_chars}


def main():
    parser = argparse.ArgumentParser(description="Download real pretraining data")
    parser.add_argument("--output", type=str, default="data/raw/",
                        help="Output directory for JSONL files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target_total_mb", type=int, default=500,
                        help="Target total data size in MB")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print(f"Downloading ~{args.target_total_mb}MB real pretraining data")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 50)

    # Proportional targets
    total = args.target_total_mb
    target_en = int(total * 0.50)   # 50% English
    target_zh_wiki = int(total * 0.35)  # 35% Chinese Wiki
    target_zh_web = int(total * 0.15)   # 15% Chinese web

    results = {}

    # 1. FineWeb-Edu English
    try:
        results["fineweb_edu"] = download_fineweb_edu(output_dir, target_mb=target_en, seed=args.seed)
    except Exception as e:
        print(f"  ❌ FineWeb-Edu failed: {e}")
        results["fineweb_edu"] = {"error": str(e)}

    # 2. Chinese Wikipedia
    try:
        results["wiki_zh"] = download_chinese_wikipedia(output_dir, target_mb=target_zh_wiki, seed=args.seed)
    except Exception as e:
        print(f"  ❌ Chinese Wikipedia failed: {e}")
        results["wiki_zh"] = {"error": str(e)}

    # 3. mc4-zh (Chinese web)
    try:
        results["mc4_zh"] = download_mc4_zh(output_dir, target_mb=target_zh_web, seed=args.seed)
    except Exception as e:
        print(f"  ❌ mc4-zh failed: {e}")
        results["mc4_zh"] = {"error": str(e)}

    # Summary
    print(f"\n{'='*50}")
    print("Download Summary")
    print(f"{'='*50}")

    total_size = 0
    total_docs = 0
    for name, r in results.items():
        if "error" in r:
            print(f"  ❌ {name}: {r['error']}")
        else:
            total_size += r["size_mb"]
            total_docs += r["count"]
            print(f"  ✅ {name}: {r['count']:,} docs, {r['size_mb']:.1f} MB")

    print(f"\n  Total: {total_docs:,} documents, {total_size:.1f} MB")
    print(f"  Files saved to: {output_dir.resolve()}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
