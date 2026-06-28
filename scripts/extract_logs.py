#!/usr/bin/env python
"""Extract loss data from training task outputs into plot-ready CSV + raw logs.

Scans all *.output files in the temp tasks dir, identifies training runs,
and saves structured CSVs for matplotlib plotting.

Output: logs/experiments/
  ├── <exp_name>/
  │   ├── raw.log          (full training output)
  │   ├── loss.csv         (step,loss,ppl columns)
  │   └── meta.json        (experiment metadata)
  └── all_experiments.csv  (combined loss for easy plotting)
"""
import re, json, os, shutil
from pathlib import Path
from datetime import datetime

TASKS_DIR = Path.home() / "AppData/Local/Temp/claude/C--Users-86136/26d1e795-a8e1-49f5-9da8-a1fc7021ed64/tasks"
OUT_DIR = Path.home() / "chat-from-scratch/logs/experiments"

# Map task IDs to experiment names
EXP_MAP = {
    "bvq8kbb6e": ("A_distill_baseline_13M_2ep", "Distill baseline first validation"),
    "bna9u08sw": ("B_minimind_13M_2ep", "MiniMind equal-token comparison"),
    "bb6lv6ybh": ("C1_distill_100pct_13M_2ep", "Distill 100% control"),
    "bz7abueby": ("C2_origwiki_20pct_13M_2ep", "Distill + original Wiki 80/20"),
    "bik07i1wn": ("C3_origwiki_40pct_13M_2ep", "Distill + original Wiki 60/40"),
    # RW-20: ran via SSH directly, log lost (only PPL known)
    "bo8hq8sk9": ("RW50_rwwiki_50pct_13M_2ep", "Distill + rewritten Wiki 50/50"),
    "bkut3cb8w": ("RW40_rwwiki_40pct_13M_2ep", "Distill + rewritten Wiki 40/60"),
    "be2kdmel5": ("E1_pure_distill_23M_2ep", "Pure distill full 23M tokens"),
    "b86306e88": ("E2_mix_rwwiki_29M_2ep", "Mixed 80/20 29M tokens"),
    "bye0mlgpf": ("E3_pure_distill_13M_8ep", "Pure distill 13M 8 epochs"),
    "b21rjvzaw": ("E4_mix_rwwiki_13M_8ep", "Mixed 80/20 13M 8 epochs"),
    "bnbfsuas4": ("S1_deep_narrow_48L_23M_2ep", "Deep & Narrow d=512 L=48 193M"),
    "bg418ec13": ("S2_wide_shallow_1024d_23M_2ep", "Wide & Shallow d=1024 L=14 207M"),
}

def extract_loss_lines(text):
    """Extract step, loss, ppl from training output lines."""
    records = []
    for line in text.split("\n"):
        # Match: "  step   150/3096 | loss=5.8270 ppl=339 | ..."
        m = re.match(r'\s+step\s+(\d+)/(\d+)\s+\|\s+loss=([\d.]+)\s+ppl=(\d+)\s+\|', line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            loss = float(m.group(3))
            ppl = int(m.group(4))
            records.append({"step": step, "total_steps": total, "loss": loss, "ppl": ppl})
    return records

def extract_meta(text):
    """Extract metadata from training header."""
    meta = {}
    # "50,000 texts → 13,350,118 tokens"
    m = re.search(r'([\d,]+)\s+texts\s+→\s+([\d,]+)\s+tokens', text)
    if m:
        meta["num_texts"] = int(m.group(1).replace(",", ""))
        meta["num_tokens"] = int(m.group(2).replace(",", ""))
    # "98,591,232 params | 3096 steps | bs=8x1024 | 2 epochs"
    m = re.search(r'([\d,]+)\s+params\s*\|\s*(\d+)\s+steps\s*\|\s*bs=(\d+)x(\d+)\s*\|\s*(\d+)\s+epochs', text)
    if m:
        meta["num_params"] = int(m.group(1).replace(",", ""))
        meta["total_steps"] = int(m.group(2))
        meta["batch_size"] = int(m.group(3))
        meta["seq_len"] = int(m.group(4))
        meta["epochs"] = int(m.group(5))
    # LR
    m = re.search(r'LR=([\d.]+)', text)
    if m:
        meta["lr"] = float(m.group(1))
    # VAL PPL
    m = re.search(r'VAL PPL:\s*([\d.]+)', text)
    if m:
        meta["val_ppl"] = float(m.group(1))
    # Training time
    m = re.search(r'DONE:.*\n.*VAL PPL:.*?\| ([\d.]+)min', text)
    if m:
        meta["train_time_min"] = float(m.group(1))
    m = re.search(r'DONE:.*\n.*\| ([\d.]+)min', text)
    if m:
        meta["train_time_min"] = float(m.group(1))
    # Data path
    m = re.search(r'SINGLE-GPU TRAINING:\s*(.+)', text)
    if m:
        meta["data_path"] = m.group(1).strip()
    return meta

# Clear and recreate output dir
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

all_records = []

for task_id, (exp_name, description) in EXP_MAP.items():
    src = TASKS_DIR / f"{task_id}.output"
    if not src.exists():
        print(f"SKIP {exp_name}: no output file")
        continue

    text = src.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        print(f"SKIP {exp_name}: empty output")
        continue

    # Check if this is actually a training run
    if "SINGLE-GPU TRAINING" not in text and "VAL PPL" not in text:
        print(f"SKIP {exp_name}: not a training run")
        continue

    exp_dir = OUT_DIR / exp_name
    exp_dir.mkdir(exist_ok=True)

    # Save raw log
    raw_path = exp_dir / "raw.log"
    raw_path.write_text(text, encoding="utf-8")

    # Extract loss
    records = extract_loss_lines(text)
    if records:
        csv_path = exp_dir / "loss.csv"
        with open(csv_path, "w") as f:
            f.write("step,total_steps,loss,ppl\n")
            for r in records:
                f.write(f"{r['step']},{r['total_steps']},{r['loss']:.4f},{r['ppl']}\n")
                r["experiment"] = exp_name
                all_records.append(r)

    # Extract meta
    meta = extract_meta(text)
    meta["experiment"] = exp_name
    meta["description"] = description
    meta["num_loss_points"] = len(records)

    # Save meta
    meta_path = exp_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"OK  {exp_name}: {len(records)} loss points | PPL={meta.get('val_ppl','N/A')}")

# Combined CSV
if all_records:
    combined_path = OUT_DIR / "all_experiments.csv"
    with open(combined_path, "w") as f:
        f.write("experiment,step,total_steps,loss,ppl\n")
        for r in all_records:
            f.write(f"{r['experiment']},{r['step']},{r['total_steps']},{r['loss']:.4f},{r['ppl']}\n")
    print(f"\nCombined: {len(all_records):,} loss points from {len(set(r['experiment'] for r in all_records))} experiments")
    print(f"Output: {OUT_DIR}")
else:
    print("\nNo loss data found. Check output files.")
