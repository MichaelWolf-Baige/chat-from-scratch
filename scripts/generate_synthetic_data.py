#!/usr/bin/env python
"""Generate synthetic Chinese-English mixed text for smoke testing.

Usage:
    python scripts/generate_synthetic_data.py --output data/raw/synthetic_train.jsonl --num_lines 10000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Templates ──────────────────────────────────────────────────────────────

TEMPLATES_EN = [
    "The {noun} is {adj}. It can {verb} very efficiently.",
    "Scientists discovered that {noun} affects {noun} in unexpected ways.",
    "A recent study shows that {adj} {noun} leads to better {noun} performance.",
    "Researchers developed a new {adj} method to {verb} {noun} with high accuracy.",
    "The {noun} system uses {adj} algorithms to {verb} data in real time.",
    "Many experts believe that {adj} {noun} will transform the future of {noun}.",
    "When you {verb} the {noun}, the {adj} results become immediately apparent.",
    "The relationship between {noun} and {noun} has been studied for decades.",
    "Using {adj} techniques, the team managed to {verb} complex {noun} problems.",
    "The {noun} framework provides {adj} tools for {noun} analysis.",
]

TEMPLATES_ZH = [
    "今天天气{adj}，非常适合去{noun}学习新知识。",
    "研究人员发现{noun}可以显著提高{noun}的处理效率。",
    "这个{adj}的{noun}方案在企业中得到了广泛应用。",
    "通过{adj}的方法，团队成功{noun}了复杂的数据结构。",
    "{noun}技术正在改变我们与{noun}互动的方式。",
    "在{noun}领域，{adj}的创新往往来自跨学科的{noun}研究。",
    "最新的{noun}系统采用了{adj}的架构设计，性能提升了{noun}。",
    "实验结果表明，{adj}的{noun}能够有效地处理大规模{noun}任务。",
    "对于{noun}来说，{adj}的数据预处理是{noun}成功的关键步骤。",
    "该{noun}平台整合了多种{adj}工具，为{noun}提供了完整解决方案。",
]

# ── Vocabulary ─────────────────────────────────────────────────────────────

NOUNS_EN = [
    "model", "system", "network", "algorithm", "database", "framework",
    "transformer", "encoder", "decoder", "classifier", "pipeline",
    "dataset", "token", "embedding", "layer", "gradient", "optimizer",
    "attention", "sequence", "language", "code", "function", "module",
]

VERBS_EN = [
    "process", "analyze", "compute", "train", "generate", "predict",
    "optimize", "transform", "evaluate", "implement", "design",
    "simulate", "extract", "classify", "detect", "enhance", "integrate",
]

ADJS_EN = [
    "efficient", "robust", "scalable", "accurate", "flexible", "reliable",
    "advanced", "modern", "powerful", "lightweight", "high-performance",
    "adaptive", "intelligent", "automated", "comprehensive", "fast",
]

NOUNS_ZH = [
    "模型", "系统", "网络", "算法", "数据库", "框架",
    "数据", "代码", "函数", "模块", "接口", "架构",
    "方法", "工具", "平台", "引擎", "服务", "应用",
]

ADJS_ZH = [
    "高效", "稳定", "灵活", "精准", "可靠", "先进",
    "强大", "轻量", "智能", "全面", "快速", "自动化",
]

VERBS_ZH = [
    "优化", "处理", "分析", "训练", "生成", "预测",
    "提升", "改善", "加速", "简化", "增强",
]


def generate_synthetic_text(num_lines: int, seed: int = 42) -> list[dict]:
    """Generate synthetic JSONL lines with mixed languages."""
    random.seed(seed)
    lines = []

    for i in range(num_lines):
        if random.random() < 0.5:
            tmpl = random.choice(TEMPLATES_EN)
            text = tmpl.format(
                noun=random.choice(NOUNS_EN),
                verb=random.choice(VERBS_EN),
                adj=random.choice(ADJS_EN),
            )
        else:
            tmpl = random.choice(TEMPLATES_ZH)
            text = tmpl.format(
                noun=random.choice(NOUNS_ZH),
                adj=random.choice(ADJS_ZH),
            ).replace("{verb}", random.choice(VERBS_ZH))

        lines.append({"text": text, "id": f"synthetic_{i:06d}"})

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data for smoke testing")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_train.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--num_lines", type=int, default=10000,
                        help="Number of lines to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Generate
    lines = generate_synthetic_text(args.num_lines, seed=args.seed)

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # Stats
    total_chars = sum(len(l["text"]) for l in lines)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    zh_lines = sum(1 for l in lines if any('一' <= c <= '鿿' for c in l["text"]))

    print(f"Generated {len(lines):,} lines")
    print(f"  Total chars: {total_chars:,}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Chinese lines: {zh_lines} (~{zh_lines/len(lines)*100:.0f}%)")
    print(f"  English lines: {len(lines)-zh_lines} (~{(len(lines)-zh_lines)/len(lines)*100:.0f}%)")
    print(f"  Saved to: {output_path.resolve()}")
    print()
    print("Sample texts:")
    for line in lines[:5]:
        print(f"  → {line['text'][:80]}")


if __name__ == "__main__":
    main()
