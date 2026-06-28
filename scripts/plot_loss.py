#!/usr/bin/env python
"""Plot training loss curves from extracted experiment data.

Usage:
    python scripts/plot_loss.py                           # plot all experiments
    python scripts/plot_loss.py -e A_distill_baseline_13M_2ep  # single experiment
    python scripts/plot_loss.py -g phase1                  # phase1 comparison
    python scripts/plot_loss.py -g scale                   # scale experiments
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless
from pathlib import Path
import argparse

LOGS_DIR = Path(__file__).parent.parent / "logs/experiments"
OUT_DIR = Path(__file__).parent.parent / "logs/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Groupings for comparison plots
GROUPS = {
    "phase1": {
        "title": "Phase 1: Data Mixing Ratios (13M tokens, 2 epochs)",
        "exps": ["C1_distill_100pct_13M_2ep", "C2_origwiki_20pct_13M_2ep",
                 "C3_origwiki_40pct_13M_2ep", "RW40_rwwiki_40pct_13M_2ep",
                 "RW50_rwwiki_50pct_13M_2ep"],
        "labels": ["100% Distill (PPL=5)", "80/20 Orig Wiki (PPL=8)",
                   "60/40 Orig Wiki (PPL=12)", "60/40 Rewritten Wiki (PPL=9)",
                   "50/50 Rewritten Wiki (PPL=10)"],
    },
    "baseline": {
        "title": "A/B Comparison: Distill vs MiniMind (13M tokens, 2 epochs)",
        "exps": ["A_distill_baseline_13M_2ep", "B_minimind_13M_2ep",
                 "C1_distill_100pct_13M_2ep"],
        "labels": ["Distill (A, PPL=5)", "MiniMind (B, PPL=11)",
                   "Distill control (C1, PPL=5)"],
    },
    "scale_data": {
        "title": "Scale Data: 13M vs 23-29M tokens (2 epochs)",
        "exps": ["C1_distill_100pct_13M_2ep", "E1_pure_distill_23M_2ep",
                 "C2_origwiki_20pct_13M_2ep", "E2_mix_rwwiki_29M_2ep"],
        "labels": ["Pure Distill 13M (PPL=5)", "Pure Distill 23M (PPL=5)",
                   "Mix Wiki 13M (PPL=8)", "Mix Wiki 29M (PPL=5)"],
    },
    "scale_epochs": {
        "title": "Scale Epochs: 2 vs 8 epochs (13M tokens)",
        "exps": ["C1_distill_100pct_13M_2ep", "E3_pure_distill_13M_8ep",
                 "E4_mix_rwwiki_13M_8ep"],
        "labels": ["Pure 2ep (PPL=5)", "Pure 8ep (PPL=6) [overfit!]",
                   "Mix 8ep (PPL=8) [overfit!]"],
    },
}

# Colorblind-friendly palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
          "#56B4E9", "#F0E442", "#000000"]

def load_experiment(name):
    csv_path = LOGS_DIR / name / "loss.csv"
    meta_path = LOGS_DIR / name / "meta.json"
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    return df, meta

def plot_single(name, ax=None):
    df, meta = load_experiment(name)
    if df is None:
        print(f"Experiment not found: {name}")
        return
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["loss"], linewidth=0.8, alpha=0.7, label="train loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name}\nPPL={meta.get('val_ppl','?')} | {meta.get('num_tokens','?')} tokens | {meta.get('epochs','?')} epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_group(group_key, ax=None, smooth=50):
    group = GROUPS[group_key]
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, label) in enumerate(zip(group["exps"], group["labels"])):
        df, meta = load_experiment(name)
        if df is None:
            print(f"  SKIP {name}: not found")
            continue
        color = COLORS[i % len(COLORS)]
        # Rolling average for smoother curves
        if len(df) > smooth:
            df["loss_smooth"] = df["loss"].rolling(smooth, center=True).mean()
            ax.plot(df["step"], df["loss_smooth"], color=color, linewidth=1.5, label=label)
        else:
            ax.plot(df["step"], df["loss"], color=color, linewidth=0.8, alpha=0.8, label=label)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(group["title"], fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="Single experiment to plot")
    parser.add_argument("-g", "--group", help="Group to plot", choices=list(GROUPS.keys()))
    parser.add_argument("--all", action="store_true", help="Plot all groups")
    args = parser.parse_args()

    if args.experiment:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_single(args.experiment, ax)
        out_path = OUT_DIR / f"{args.experiment}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()
        return

    groups_to_plot = list(GROUPS.keys()) if args.all else ([args.group] if args.group else ["baseline", "phase1", "scale_data", "scale_epochs"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for gk, ax in zip(groups_to_plot[:4], axes.flat):
        plot_group(gk, ax)

    if len(groups_to_plot) > 4:
        for gk in groups_to_plot[4:]:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            plot_group(gk, ax2)
            out_path = OUT_DIR / f"{gk}.png"
            fig2.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved: {out_path}")

    fig.tight_layout()
    out_path = OUT_DIR / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
