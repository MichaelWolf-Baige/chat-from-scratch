#!/usr/bin/env python
"""Distillation pipeline: Qwen2.5-1.5B → pretraining data.

Mimics MiniMind's data strategy using teacher model distillation.
Optimized for 3090: torch.compile + left padding + batched generation.

Per-card: ~25K texts in ~1h on single RTX 3090
4 cards parallel: ~100K texts in ~1h

Usage:
  # Single card test (100 samples)
  CUDA_VISIBLE_DEVICES=0 python scripts/distill_pipeline.py --n 100 -o data/test.jsonl

  # 4-card parallel (each generates 25K independently)
  CUDA_VISIBLE_DEVICES=0 python scripts/distill_pipeline.py --n 25000 -o data/distill_0.jsonl &
  CUDA_VISIBLE_DEVICES=1 python scripts/distill_pipeline.py --n 25000 -o data/distill_1.jsonl &
  CUDA_VISIBLE_DEVICES=2 python scripts/distill_pipeline.py --n 25000 -o data/distill_2.jsonl &
  CUDA_VISIBLE_DEVICES=3 python scripts/distill_pipeline.py --n 25000 -o data/distill_3.jsonl &
  wait && cat data/distill_*.jsonl > data/distill_100k.jsonl
"""

import os, sys, argparse, json, time, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"  # Use cached model, no network needed

# ═══════════════════════════════════════════════════════════════
# SEED PROMPTS — 2000+ diverse Chinese prompts across 8 domains
# ═══════════════════════════════════════════════════════════════

SEEDS = []

# --- Encyclopedia (350 prompts) ---
TOPICS = [
    "人工智能","机器学习","深度学习","自然语言处理","计算机视觉","神经网络",
    "量子计算","区块链","云计算","物联网","5G通信","边缘计算",
    "基因编辑","光合作用","黑洞","相对论","量子力学","暗物质",
    "细胞分裂","生态系统","气候变化","可再生能源","碳中和","生物多样性",
    "中国历史","世界历史","文艺复兴","工业革命","丝绸之路","四大发明",
    "市场经济","货币政策","国际贸易","数字经济","共享经济","通货膨胀",
    "神经系统","免疫系统","DNA结构","疫苗原理","抗生素","蛋白质",
    "太阳系","银河系","超新星","行星运动","地壳运动","大气层",
]
for t in TOPICS:
    SEEDS.append(f"请用中文写一段关于{t}的科普介绍，约100-200字，语言通俗易懂、内容准确。")
    SEEDS.append(f"什么是{t}？请用中文简洁解释，并举一个日常生活中的例子。")
    SEEDS.append(f"以{t}为主题写一篇中文说明文，适合高中生阅读。要求结构清晰。")
    SEEDS.append(f"用一问一答的对话形式介绍{t}的基本概念。")
    SEEDS.append(f"列出关于{t}的5个中文趣味知识点，每条20字左右。")

# --- Story / Narrative (300 prompts) ---
ELEMENTS = [
    ("小明","森林","勇敢"), ("小红","海边","友谊"), ("一只流浪猫","城市","善良"),
    ("老爷爷","山村","传承"), ("宇航员","火星","坚持"), ("程序员","深夜办公室","创造"),
    ("少女","魔法学校","成长"), ("机器人","未来世界","自由"), ("船长","暴风雨","责任"),
    ("画家","古镇","灵感"), ("小狐狸","雪原","信任"), ("小男孩","图书馆","好奇"),
]
for who, where, what in ELEMENTS:
    SEEDS.append(f"写一个中文短故事，主角是{who}，地点在{where}，主题关于{what}。约150-200字。")
    SEEDS.append(f"以{who}的第一人称视角，讲述在{where}发生的一段经历，要有情感描写。")
    SEEDS.append(f"创作一个寓言故事，角色包括{who}，地点在{where}，通过故事传达关于{what}的道理。")

for i in range(50):
    SEEDS.append("请写一个简短的中文故事，有完整的情节，结局要温暖人心。约150字。")

# --- Code / Technical (300 prompts) ---
CODE_T = [
    "Python列表推导式","递归函数原理","SQL的JOIN操作","Git分支管理","REST API设计",
    "Docker容器","MVC架构","TCP和UDP的区别","快速排序算法","二叉树遍历",
    "正则表达式","面向对象三大特性","数据库索引","HTTP和HTTPS","WebSocket",
]
for ct in CODE_T:
    SEEDS.append(f"用中文解释{ct}，给出说明和代码示例。")
    SEEDS.append(f"写一篇面向初学者的中文教程，介绍{ct}的基本用法。")

for i in range(80):
    SEEDS.append("写一个实用的Python函数，包含详细的中文注释，解决一个常见问题。")

# --- Dialogue / QA (300 prompts) ---
for i in range(100):
    roles = [("顾客","客服"), ("学生","老师"), ("朋友","朋友"), ("面试官","求职者"), ("医生","患者")]
    r1, r2 = roles[i % 5]
    SEEDS.append(f"模拟一段{r1}和{r2}的中文对话，{'咨询' if i%3==0 else '闲聊' if i%3==1 else '解决问题'}，自然真实。")
    SEEDS.append("写一段关于" + ["学习方法","职业规划","健康饮食","环保生活","科技趋势"][i%5] + "的中文问答对话。")

# --- News / Article (250 prompts) ---
NEWS_T = ["科技突破","环保成就","教育改革","医疗进步","体育赛事","文化活动"]
for nt in NEWS_T:
    SEEDS.append(f"写一篇关于{nt}的模拟中文新闻报道，含标题、导语和正文。约150字。")
    SEEDS.append(f"以记者视角写关于{nt}的中文深度分析，含原因和影响。约200字。")

for i in range(100):
    SEEDS.append("写一篇简短的中文评论文章，讨论一个社会热点，表达明确的观点。约150字。")

# --- Poetry / Creative (200 prompts) ---
for i in range(50):
    theme = ["春天","秋天","月夜","大海","故乡","母爱","友谊","时光"][i%8]
    SEEDS.append(f"写一首关于{theme}的中文现代诗，简洁优美。")
    SEEDS.append("用中文写一段优美的散文，描写" + ["清晨山林","黄昏海边","雨后街道","星空下","雪中村落"][i%5] + "。")
    SEEDS.append("写一首简短的中文五言绝句，题目自拟。")

# --- How-to (200 prompts) ---
HOWTO = ["番茄炒蛋","煮咖啡","写求职信","打包行李","养多肉","学外语",
         "提高写作","健康作息","布置书桌","拍好照片","管理时间","克服拖延"]
for ht in HOWTO:
    SEEDS.append(f"用中文写详细的步骤指南：如何{ht}。要有实操性。约150字。")

for i in range(100):
    SEEDS.append("写一份简短的实用中文生活指南，介绍一个技能或技巧。")

# --- Reasoning (200 prompts) ---
for i in range(100):
    SEEDS.append("提出一个生活中的两难问题，用中文进行逻辑分析并给出解决方案。约150字。")
    SEEDS.append("用中文写一段推理分析：从已知条件出发，逐步推理得出结论。逻辑要清晰。")

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_samples", type=int, default=100)
    parser.add_argument("-o", "--output", default="data/distilled.jsonl")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Distill Pipeline: {args.n_samples} samples | batch={args.batch_size} | seed={args.seed}")

    # ── Load Teacher ──
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda:0", local_files_only=True,
        attn_implementation="sdpa",
    ).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Teacher: {n_params:,} params")

    # ── Select prompts ──
    prompts = random.sample(SEEDS, min(args.n_samples, len(SEEDS)))
    while len(prompts) < args.n_samples:
        prompts.extend(random.sample(SEEDS, min(args.n_samples - len(prompts), len(SEEDS))))
    print(f"  Prompts: {len(prompts):,}")

    # ── Generate ──
    all_results = []
    t0 = time.time()

    for b_start in range(0, len(prompts), args.batch_size):
        batch = prompts[b_start:b_start + args.batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=256).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.8, top_p=0.9,
                do_sample=True, pad_token_id=tok.eos_token_id,
            )

        for j, out in enumerate(outputs):
            in_len = inputs["input_ids"][j].shape[0]
            text = tok.decode(out[in_len:], skip_special_tokens=True).strip()
            if len(text) >= 30:
                all_results.append({"text": text})

        b_num = b_start // args.batch_size + 1
        total_b = (len(prompts) + args.batch_size - 1) // args.batch_size

        if b_num <= 5 or b_num % 50 == 0:
            elapsed = time.time() - t0
            chars = sum(len(r["text"]) for r in all_results)
            print(f"  [{b_num:4d}/{total_b}] {(b_num/ total_b*100):5.1f}% | "
                  f"{len(all_results):,} texts | {chars:,} chars | "
                  f"{chars/elapsed:.0f} c/s | {elapsed/60:.0f}min")

    # ── Save ──
    with open(args.output, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    total_c = sum(len(r["text"]) for r in all_results)
    size_mb = os.path.getsize(args.output) / 1e6

    print(f"\nDone! {len(all_results):,} texts | {total_c:,} chars | {size_mb:.1f}MB")
    print(f"Time: {elapsed/60:.0f}min | {elapsed/len(all_results):.1f}s/text | {total_c/elapsed:.0f} c/s")
    print(f"Output: {args.output}")

    # Show samples
    for r in random.sample(all_results, min(2, len(all_results))):
        print(f"\n  [{len(r['text'])} chars]: {r['text'][:150]}...")


if __name__ == "__main__":
    main()
