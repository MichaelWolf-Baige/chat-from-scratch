#!/usr/bin/env python
"""Build mixed datasets with configurable template/real data ratios.

Creates multiple datasets for ablation experiments:
  Ratio 100/0: Pure template (baseline)
  Ratio 90/10: 90% template + 10% real
  Ratio 80/20: 80% template + 20% real
  Ratio 70/30: 70% template + 30% real
  Ratio 50/50: 50% template + 50% real

Each mixed dataset is written as a separate JSONL file ready for training.

Usage:
  python scripts/build_mixed_dataset.py --num_total 100000 --template_file data/pretrain/template_150k.jsonl --real_dir data/real/

Output:
  data/mixed/ratio_100_0.jsonl  (100% template)
  data/mixed/ratio_90_10.jsonl  (90% template + 10% real)
  data/mixed/ratio_80_20.jsonl  (80% template + 20% real)
  data/mixed/ratio_70_30.jsonl  (70% template + 30% real)
  data/mixed/ratio_50_50.jsonl  (50% template + 50% real)
"""

import argparse, json, random, sys
from pathlib import Path
from collections import Counter

SEED = 42

def load_jsonl(filepath: str | Path) -> list[dict]:
    """Load all lines from a JSONL file."""
    texts = []
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  WARNING: {filepath} not found, skipping")
        return texts
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                texts.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return texts


def load_directory(dirpath: str | Path) -> list[dict]:
    """Load all JSONL files from a directory."""
    all_texts = []
    dirpath = Path(dirpath)
    if not dirpath.exists():
        print(f"  WARNING: {dirpath} not found")
        return all_texts
    for filepath in sorted(dirpath.glob("*.jsonl")):
        loaded = load_jsonl(filepath)
        all_texts.extend(loaded)
        print(f"  Loaded {len(loaded):,} from {filepath.name}")
    return all_texts


def build_mixed(template_texts: list[dict], real_texts: list[dict],
                ratio: float, num_total: int, seed: int) -> list[dict]:
    """Build a mixed dataset at the given ratio of template data.

    Args:
        template_texts: List of template-generated {"text": ...} dicts
        real_texts: List of real-world {"text": ...} dicts
        ratio: Fraction of template data (0.0 to 1.0)
        num_total: Target total number of texts
        seed: Random seed for reproducibility

    Returns:
        Shuffled mixed dataset of length num_total (or less if insufficient data)
    """
    rng = random.Random(seed)

    n_template = int(num_total * ratio)
    n_real = num_total - n_template

    # Sample with replacement if not enough data
    if len(template_texts) >= n_template:
        templates = rng.sample(template_texts, n_template)
    else:
        templates = rng.choices(template_texts, k=n_template)

    if len(real_texts) >= n_real:
        reals = rng.sample(real_texts, n_real)
    else:
        reals = rng.choices(real_texts, k=n_real)

    # Add source labels
    for t in templates:
        t["source"] = "template"
    for r in reals:
        if "source" not in r:
            r["source"] = "real"

    mixed = templates + reals
    rng.shuffle(mixed)
    return mixed


def main():
    parser = argparse.ArgumentParser(description="Build mixed datasets for ablation")
    parser.add_argument("--num_total", type=int, default=100000,
                       help="Total texts per mixed dataset (default: 100000)")
    parser.add_argument("--template_file", type=str,
                       default="data/pretrain/template_150k.jsonl",
                       help="Template data JSONL file")
    parser.add_argument("--real_dir", type=str, default="data/real",
                       help="Directory containing real data JSONL files")
    parser.add_argument("--output_dir", type=str, default="data/mixed",
                       help="Output directory for mixed datasets")
    parser.add_argument("--ratios", type=str, default="100/0,90/10,80/20,70/30,50/50",
                       help="Comma-separated ratios (template/real)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building Mixed Datasets (Ablation Experiment)")
    print(f"  Target per dataset: {args.num_total:,} texts")
    print(f"  Ratios: {args.ratios}")
    print("=" * 60)

    # Load data sources
    print("\n[1/3] Loading template data...")
    template_texts = load_jsonl(args.template_file)
    if not template_texts:
        # Try directory
        template_texts = load_directory(Path(args.template_file).parent)
    print(f"  Template: {len(template_texts):,} texts loaded")

    print("\n[2/3] Loading real data...")
    real_texts = load_directory(args.real_dir)
    print(f"  Real: {len(real_texts):,} texts loaded")

    # Build mixed datasets
    print(f"\n[3/3] Building {len(args.ratios.split(','))} mixed datasets...")
    ratios_parsed = []
    for r_str in args.ratios.split(","):
        parts = r_str.strip().split("/")
        template_pct = int(parts[0])
        real_pct = int(parts[1])
        ratio = template_pct / 100.0
        ratios_parsed.append((template_pct, real_pct, ratio))

    for template_pct, real_pct, ratio in ratios_parsed:
        label = f"ratio_{template_pct}_{real_pct}"
        output_file = output_dir / f"{label}.jsonl"

        print(f"\n  Building {label} ({template_pct}/{real_pct})...")
        mixed = build_mixed(template_texts, real_texts, ratio,
                          args.num_total, seed=args.seed + template_pct)

        # Write
        with open(output_file, "w", encoding="utf-8") as f:
            for item in mixed:
                f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")

        # Stats
        sources = Counter(item.get("source", "unknown") for item in mixed)
        lengths = [len(item["text"]) for item in mixed]
        total_chars = sum(lengths)

        print(f"    Written: {len(mixed):,} texts | {total_chars/1024/1024:.1f} MB")
        print(f"    Length: mean={sum(lengths)/len(lengths):.0f} median={sorted(lengths)[len(lengths)//2]}")
        print(f"    Sources: {dict(sources)}")
        print(f"    -> {output_file}")

    print(f"\n{'=' * 60}")
    print(f"ABLATION DATASETS READY in {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
