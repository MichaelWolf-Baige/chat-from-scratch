#!/usr/bin/env python
"""分析 benchmark 结果，生成对比报告。

用法:
    python scripts/analyze_benchmark.py logs/benchmarks/p3_ours.json logs/benchmarks/cap_wiki_100M.json ...
"""

import json, sys
from pathlib import Path
from collections import defaultdict


def load_results(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze(all_runs: list[dict]):
    """分析所有 benchmark 结果并打印对比报告。"""

    by_ckpt = defaultdict(lambda: defaultdict(dict))
    ckpt_names = set()
    bench_names = set()
    dec_names = set()

    for run in all_runs:
        ckpt = run["checkpoint"]["name"]
        bench = run["benchmark"]
        dec = run["decoding"]
        by_ckpt[ckpt][bench][dec] = run
        ckpt_names.add(ckpt)
        bench_names.add(bench)
        dec_names.add(dec)

    ckpt_names = sorted(ckpt_names)
    bench_names = sorted(bench_names)
    dec_names = sorted(dec_names)

    print("=" * 80)
    print("Benchmark 对比分析报告")
    print("=" * 80)

    # 1. 续写质量对比
    print("\n" + "-" * 50)
    print("1. 续写式 (Completion) — 测预训练原始能力")
    print("-" * 50)

    for ckpt in ckpt_names:
        if "completion" not in by_ckpt[ckpt]:
            continue
        if "default" not in by_ckpt[ckpt]["completion"]:
            continue

        run = by_ckpt[ckpt]["completion"]["default"]
        meta = run["checkpoint"]
        print(f"\n### {ckpt} (PPL={meta['val_ppl']}, {meta['params']:,} params, {meta['steps']} steps)")

        for gen in run["generations"]:
            prompt = gen["prompt"]
            resp = gen["generation"][:120]
            print(f"  [{gen['category']}] {prompt}...")
            print(f"        -> {resp}")
            print()

    # 2. 对话质量对比
    print("\n" + "-" * 50)
    print("2. 对话式 (Dialogue) — 匹配训练格式的生成")
    print("-" * 50)

    for ckpt in ckpt_names:
        if "dialogue" not in by_ckpt[ckpt]:
            continue
        if "default" not in by_ckpt[ckpt]["dialogue"]:
            continue

        run = by_ckpt[ckpt]["dialogue"]["default"]
        meta = run["checkpoint"]
        print(f"\n### {ckpt} (PPL={meta['val_ppl']}, {meta['params']:,} params)")

        for gen in run["generations"]:
            resp = gen["generation"][:150]
            avg_tok = gen.get("tokens", 0)
            print(f"  [{gen['category']}] {gen['prompt'][:50].strip()}")
            print(f"        -> ({avg_tok} tok) {resp}")
            print()

    # 3. 聊天格式退化测试
    print("\n" + "-" * 50)
    print("3. 聊天式 (Chat) — 格式不匹配时的退化程度")
    print("-" * 50)

    for ckpt in ckpt_names:
        if "chat" not in by_ckpt[ckpt]:
            continue
        if "default" not in by_ckpt[ckpt]["chat"]:
            continue

        run = by_ckpt[ckpt]["chat"]["default"]
        meta = run["checkpoint"]
        print(f"\n### {ckpt} (PPL={meta['val_ppl']})")

        for gen in run["generations"]:
            resp = gen["generation"][:120]
            print(f"  [{gen['category']}] Q: {gen['prompt'][:40]}")
            print(f"        A: {resp}")
            print()

    # 4. 解码参数灵敏度
    print("\n" + "-" * 50)
    print("4. 解码参数灵敏度 — 不同 temperature 下的多样性变化")
    print("-" * 50)

    for bench in bench_names:
        print(f"\n### Benchmark: {bench}")
        for ckpt in ckpt_names:
            if bench not in by_ckpt[ckpt]:
                continue
            print(f"  {ckpt}:")
            for dec in dec_names:
                if dec not in by_ckpt[ckpt][bench]:
                    continue
                run = by_ckpt[ckpt][bench][dec]
                gens = run["generations"]
                avg_tokens = sum(g.get("tokens", 0) for g in gens) / max(len(gens), 1)
                openings = [g["generation"][:30] for g in gens if g["generation"]]
                unique_openings = len(set(openings))
                diversity = unique_openings / max(len(openings), 1) * 100
                print(f"    {dec:15s}: avg_tok={avg_tokens:.0f}, diversity={diversity:.0f}%")

    # 5. PPL vs 生成质量相关性
    print("\n" + "-" * 50)
    print("5. PPL vs 生成质量 — 相关性检验")
    print("-" * 50)
    print(f"{'Checkpoint':<25s} {'PPL':>6s} {'Dialog Avg Tok':>14s} {'Comp Diversity':>14s} {'Dialog Diversity':>15s}")
    print("-" * 75)

    for ckpt in ckpt_names:
        if "dialogue" not in by_ckpt[ckpt]:
            continue
        if "default" not in by_ckpt[ckpt]["dialogue"]:
            continue

        meta = by_ckpt[ckpt]["dialogue"]["default"]["checkpoint"]
        ppl = meta["val_ppl"]

        dialog_gens = by_ckpt[ckpt]["dialogue"]["default"]["generations"]
        dialog_tokens = sum(g.get("tokens", 0) for g in dialog_gens) / max(len(dialog_gens), 1)

        if "completion" in by_ckpt[ckpt] and "default" in by_ckpt[ckpt]["completion"]:
            comp_gens = by_ckpt[ckpt]["completion"]["default"]["generations"]
            comp_openings = [g["generation"][:30] for g in comp_gens if g["generation"]]
            comp_div = len(set(comp_openings)) / max(len(comp_openings), 1) * 100
        else:
            comp_div = 0

        dialog_openings = [g["generation"][:30] for g in dialog_gens if g["generation"]]
        dialog_div = len(set(dialog_openings)) / max(len(dialog_openings), 1) * 100

        print(f"{ckpt:<25s} {ppl:>6.1f} {dialog_tokens:>14.0f} {comp_div:>13.0f}% {dialog_div:>14.0f}%")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_benchmark.py <result1.json> [result2.json ...]")
        sys.exit(1)

    all_runs = []
    for path in sys.argv[1:]:
        try:
            runs = load_results(path)
            all_runs.extend(runs)
            print(f"加载: {path} ({len(runs)} 个运行)")
        except Exception as e:
            print(f"加载失败 {path}: {e}")

    if not all_runs:
        print("没有数据可分析")
        sys.exit(1)

    print()
    analyze(all_runs)


if __name__ == "__main__":
    main()
