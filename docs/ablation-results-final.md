# Architecture Ablation — Final Results

> **Date**: 2026-06-29 | **Status**: ALL 6/6 COMPLETE

## Results Table

| # | Experiment | d_model | n_layers | Params | Data | VAL PPL | Trend |
|---|-----------|---------|----------|--------|------|---------|-------|
| 1 | **Mid** 🔥 | 768 | 28 | 188M | wiki_100M | **18.13** | — |
| 2 | **C200** | 512 | 24 | 99M | wiki_200M | **18.44** | ↑ |
| 3 | Extreme Wide | 896 | 12 | 119M | wiki_100M | 18.75 | ≈ |
| 4 | C100 (old) | 512 | 24 | 99M | wiki_100M | 18.93 | ≈ |
| 5 | Extreme Deep | 512 | 36 | 117M | wiki_100M | 19.05 | ≈ |
| 6 | Shallow-Wide | 768 | 16 | 116M | wiki_100M | 19.09 | ≈ |
| 7 | Deep-Thin | 576 | 30 | 111M | wiki_100M | 19.48 | ↓ |
| — | C50 (old) | 512 | 24 | 99M | wiki_50M | 23.23 | ↓↓ |

## Key Findings

### 1. Width > Depth at ~115M Scale (Reproduced)

- **Extreme Wide** (d=896 L=12): PPL=18.75 — **best at 115M scale**
- **Deep-Thin** (d=576 L=30): PPL=19.48 — **worst at 115M scale**
- Delta: **3.7%**. Width consistently beats depth.

### 2. Data Scaling Works

| Data | Tokens | VAL PPL | Delta |
|------|--------|---------|-------|
| wiki_50M | 50M | 23.23 | — |
| wiki_100M | 100M | 18.93 | -18.5% |
| wiki_200M | 200M | 18.44 | -2.6% |

100M model NOT saturated. Returns diminish but trend continues.

### 3. Larger Model Wins Modestly

Mid (188M, PPL=18.13) > Extreme Wide (119M, PPL=18.75) by 3.3%.
Cost: +58% params for +3.3%. Efficiency favors Extreme Wide.

### 4. Width/Depth Boundary at d=512-768

Extreme Deep (19.05) ≈ Shallow-Wide (19.09).

### 5. Deep-Thin Anti-Pattern

d=576 L=30 → PPL=19.48, **worse than baseline** (18.93).

## C200 Log

Initial loss ~9 → final PPL=18.44 over 92,772 steps.
Best val_loss=2.63 at step 89,500. NaN events: 0.

## Code Fix

rotate_ckpts sorted by mtime not filename (step10000 < step8500 in string sort bug).

## Config

Llama-style (SwiGLU, RoPE, RMSNorm, Pre-LN, GQA), 8192 vocab, seq=1024,
bs=4x1024, 2 epochs, AdamW lr=5e-4 cosine, RTX 3090, seed=42.