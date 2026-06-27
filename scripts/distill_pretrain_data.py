#!/usr/bin/env python
"""Distillation pipeline: Use Qwen2.5-1.5B as Teacher to generate pretraining data.

Mimics MiniMind's data strategy: diverse Chinese prompts → Teacher generates high-quality text.

Usage:
    # Quick test (100 samples)
    CUDA_VISIBLE_DEVICES=0 python scripts/distill_pretrain_data.py --n_samples 100

    # Full generation (100K samples, ~100MB text, ~2-3 hours on 3090)
    CUDA_VISIBLE_DEVICES=0 python scripts/distill_pretrain_data.py --n_samples 100000 --output data/distilled_pretrain.jsonl

    # Ultra batch (use ≥2 GPUs, each generates independently)
    CUDA_VISIBLE_DEVICES=0 python scripts/distill_pretrain_data.py --n_samples 50000 --output data/distilled_0.jsonl &
    CUDA_VISIBLE_DEVICES=1 python scripts/distill_pretrain_data.py --n_samples 50000 --output data/distilled_1.jsonl &
"""

import os, sys, argparse, json, time, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ═══════════════════════════════════════════════════════════════
# 2000+ seed prompts across 8 domains
# ═══════════════════════════════════════════════════════════════

SEED_PROMPTS = []

# Domain 1: Encyclopedia / Knowledge (350 prompts)
KNOWLEDGE_TOPICS = [
    "人工智能","机器学习","深度学习","自然语言处理","计算机视觉",
    "量子计算","区块链","云计算","物联网","5G通信",
    "基因编辑","光合作用","黑洞","相对论","量子力学",
    "细胞分裂","生态系统","气候变化","可再生能源","碳中和",
    "中国历史","世界历史","文艺复兴","工业革命","丝绸之路",
    "市场经济","货币政策","国际贸易","数字经济","共享经济",
    "神经系统","免疫系统","DNA结构","疫苗原理","抗生素",
    "太阳系","银河系","宇宙膨胀","暗物质","超新星",
]
for t in KNOWLEDGE_TOPICS:
    SEED_PROMPTS.append(f"请用中文写一段关于{t}的科普介绍，大约150字，语言通俗易懂。")
    SEED_PROMPTS.append(f"什么是{t}？请用简洁的中文解释，并举一个生活中的例子。")
    SEED_PROMPTS.append(f"写一段关于{t}的中文学术性介绍，适合高中以上文化程度的读者。")
    SEED_PROMPTS.append(f"用对话的方式（一问一答）介绍{t}的基本概念。")
    SEED_PROMPTS.append(f"请列出关于{t}的5个有趣的事实，用中文描述。")
    SEED_PROMPTS.append(f"以{t}为主题，写一篇简短的说明文，要求结构清晰、语言准确。")
    SEED_PROMPTS.append(f"解释{t}在日常生活中的实际应用，举三个具体例子。")

# Domain 2: Story / Narrative (300 prompts)
STORY_ELEMENTS = [
    ("小明","森林","冒险"),("小红","海边","发现"),("一只小猫","城市","奇遇"),
    ("老爷爷","山村","回忆"),("宇航员","火星","探索"),("程序员","深夜办公室","灵感"),
    ("画家","巴黎街头","邂逅"),("少女","魔法学校","成长"),("侦探","雨夜","调查"),
    ("机器人","未来世界","觉醒"),("小女孩","花园","秘密"),("船长","暴风雨","勇气"),
]
for who, where, what in STORY_ELEMENTS:
    SEED_PROMPTS.append(f"写一个中文短故事，主角是{who}，地点在{where}，主题是关于{what}。约200字。")
    SEED_PROMPTS.append(f"以{who}的视角，描述一个在{where}发生的故事，结局要温暖感人。")
    SEED_PROMPTS.append(f"写一个寓言故事，角色包括{who}，地点在{where}，通过故事说明一个道理。")

for i in range(50):
    SEED_PROMPTS.append(f"请写一个简短的中文故事，主题随机，要有完整的情节起承转合，约150字。")

# Domain 3: Code / Technical (300 prompts)
CODE_TOPICS = [
    "Python的列表推导式","如何用Python读取CSV文件","递归函数的原理",
    "SQL的JOIN操作","Git的分支管理","REST API的设计原则",
    "Docker的基本使用方法","什么是MVC架构","TCP和UDP的区别",
    "排序算法中的快速排序","二叉树的遍历","正则表达式的常用语法",
    "面向对象编程的三大特性","函数式编程和命令式编程的区别",
    "数据库索引的工作原理","HTTP和HTTPS的区别","WebSocket的使用场景",
]
for ct in CODE_TOPICS:
    SEED_PROMPTS.append(f"请用中文解释{ct}，给出清晰的说明和代码示例。")
    SEED_PROMPTS.append(f"写一篇面向初学者的中文教程，介绍{ct}的基本用法。")

for i in range(80):
    SEED_PROMPTS.append(f"写一个实用的Python代码片段，包含详细的中文注释，解决一个常见的编程问题。")

# Domain 4: Dialogue / QA (300 prompts)
for i in range(100):
    SEED_PROMPTS.append(f"模拟一段中文客服对话，顾客咨询一个常见问题，客服耐心解答。")
    SEED_PROMPTS.append(f"写一段老师和学生关于{'数学' if i%3==0 else '物理' if i%3==1 else '编程'}的一问一答教学对话。")
    SEED_PROMPTS.append(f"模拟一段朋友之间的中文闲聊对话，话题自然、语气轻松。")

# Domain 5: News / Article (250 prompts)
NEWS_TOPICS = ["科技突破","环保成就","教育改革","医疗进步","体育赛事","文化活动","经济动态","国际合作"]
for nt in NEWS_TOPICS:
    SEED_PROMPTS.append(f"写一篇关于{nt}的模拟中文新闻报道，包括标题、导语和正文。约150字。")
    SEED_PROMPTS.append(f"以记者的视角，写一篇关于{nt}的中文深度报道，分析原因和影响。")

for i in range(100):
    SEED_PROMPTS.append(f"写一篇简短的中文评论文章，讨论一个社会热点话题，表达明确的观点。")

# Domain 6: Poetry / Creative Writing (200 prompts)
for i in range(50):
    SEED_PROMPTS.append(f"写一首关于{'春天' if i%4==0 else '秋天' if i%4==1 else '月夜' if i%4==2 else '大海'}的简短中文现代诗。")
    SEED_PROMPTS.append(f"创作一段优美的中文散文，描述{'清晨的山林' if i%3==0 else '黄昏的海边' if i%3==1 else '雨后的街道'}。")
    SEED_PROMPTS.append(f"用中文写一段富有诗意的文字，主题可以是自然、情感或人生感悟。")
    SEED_PROMPTS.append(f"写一首简短的中文古诗（五言或七言绝句），题目自拟。")

# Domain 7: How-to / Instructions (200 prompts)
HOWTO_TOPICS = ["做番茄炒蛋","煮咖啡","写一封求职邮件","打包行李","养一盆多肉植物",
                "学习一门外语","提高写作能力","保持健康作息","布置书桌","拍摄好照片"]
for ht in HOWTO_TOPICS:
    SEED_PROMPTS.append(f"用中文写一份详细的步骤指南，教人如何{ht}。内容要实用、清晰。")

for i in range(100):
    SEED_PROMPTS.append(f"写一份简短的实用中文指南，介绍一个生活技能或学习方法。")

# Domain 8: Reasoning / Logic (200 prompts)
for i in range(100):
    SEED_PROMPTS.append(f"提出一个生活中的两难问题，然后用中文进行逻辑分析，给出合理的解决方案。")
    SEED_PROMPTS.append(f"用中文写一段推理分析，从已知条件出发，逐步得出结论。要求逻辑清晰。")

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# BATCH GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════

def batch_generate(model, tokenizer, prompts, batch_size=8, max_new_tokens=200,
                   temperature=0.8, top_p=0.9):
    """Generate outputs for a list of prompts in batches."""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        # Tokenize with padding
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated_ids = output[input_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if len(text) >= 30:  # Filter too-short outputs
                results.append({"text": text.strip()})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100,
                       help="Number of samples to generate (100 for quick test, 100000 for full)")
    parser.add_argument("--output", type=str, default="data/distilled_pretrain.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for generation")
    parser.add_argument("--model_id", type=str,
                       default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Teacher model ID")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print("=" * 55)
    print(f"Distillation Pipeline: Teacher={args.model_id}")
    print(f"Target: {args.n_samples:,} samples")
    print(f"Seed prompts available: {len(SEED_PROMPTS):,}")
    print("=" * 55)

    # ── Load model ──
    print("Loading teacher model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "left"  # Required for batched decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
    ).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Teacher: {n_params:,} params")
    print(f"  GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB allocated")

    # ── Select prompts ──
    selected = random.sample(SEED_PROMPTS, min(args.n_samples, len(SEED_PROMPTS)))
    # If we need more than available, repeat with variation
    while len(selected) < args.n_samples:
        remaining = args.n_samples - len(selected)
        more = random.sample(SEED_PROMPTS, min(remaining, len(SEED_PROMPTS)))
        selected.extend(more)

    print(f"  Selected {len(selected):,} prompts for generation")

    # ── Generate ──
    all_results = []
    t0 = time.time()
    total_batches = (len(selected) + args.batch_size - 1) // args.batch_size

    print(f"\n  Batch size: {args.batch_size}")
    print(f"  Total batches: {total_batches:,}")
    print(f"  Estimated time: ~{total_batches * 8 / 60:.0f} min ({total_batches * 8 / 3600:.1f} hr)")
    print(f"  Start: {time.strftime('%H:%M:%S')}")

    for batch_start in range(0, len(selected), args.batch_size):
        batch_prompts = selected[batch_start:batch_start + args.batch_size]
        batch_results = batch_generate(
            model, tokenizer, batch_prompts,
            batch_size=args.batch_size, max_new_tokens=200,
        )
        all_results.extend(batch_results)

        # Progress
        n_done = min(batch_start + args.batch_size, len(selected))
        pct = n_done / len(selected) * 100
        elapsed = time.time() - t0
        total_tokens = sum(len(r["text"]) for r in all_results)
        rate = total_tokens / max(elapsed, 0.01)

        batch_num = batch_start // args.batch_size + 1
        if batch_num <= 5 or batch_num % 50 == 0:
            print(f"  [{batch_num:5d}/{total_batches}] {pct:5.1f}% | "
                  f"{len(all_results):,} texts | {total_tokens:,} chars | "
                  f"{rate:.0f} char/s | {elapsed/60:.0f}min")

        # Save intermediate every 200 batches
        if batch_num % 200 == 0:
            tmp_path = args.output + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for r in all_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Final save ──
    with open(args.output, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    total_chars = sum(len(r["text"]) for r in all_results)
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print(f"\n{'=' * 55}")
    print(f"Distillation Complete!")
    print(f"  Samples: {len(all_results):,}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
    print(f"  Avg: {elapsed/len(all_results):.1f}s per sample")
    print(f"  Output: {args.output}")
    print(f"{'=' * 55}")

    # Show samples
    print(f"\nSample outputs:")
    for r in random.sample(all_results, min(3, len(all_results))):
        print(f"  [{len(r['text'])} chars] {r['text'][:120]}...")
        print()


if __name__ == "__main__":
    main()
