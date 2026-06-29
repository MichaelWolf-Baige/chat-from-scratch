#!/usr/bin/env python
"""综合 Benchmark 评估脚本

功能：
  1. 加载多个 checkpoint
  2. 对 3 种 prompt 格式（续写/对话/聊天）分别评估
  3. 支持多组解码参数 sweep
  4. 输出结构化 JSON 结果

用法:
    # 评估单个 checkpoint
    python scripts/eval_benchmark.py -c checkpoints/p3_ours.pt

    # 评估多个 checkpoint，多组解码参数
    python scripts/eval_benchmark.py -c ckpt1.pt ckpt2.pt --sweep

    # 只评估特定 benchmark
    python scripts/eval_benchmark.py -c ckpt1.pt --benchmarks completion dialogue
"""

import sys, os, argparse, json, time
from pathlib import Path
from datetime import datetime

# 项目根目录
_SCRIPT_PATH = Path(os.path.abspath(__file__)) if '__file__' in globals() else Path(os.getcwd())
PROJ_ROOT = _SCRIPT_PATH.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import torch
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

# ─── 架构参数表 ─────────────────────────────────────────
ARCH_CONFIGS = {
    "100M":       dict(d_model=512,  n_layers=24, n_heads=8,  n_kv_heads=4,  d_ff=2048),
    "deep_thin":  dict(d_model=576,  n_layers=30, n_heads=9,  n_kv_heads=3,  d_ff=1536),
    "shallow_wide": dict(d_model=768,  n_layers=16, n_heads=12, n_kv_heads=4,  d_ff=2304),
    "extreme_deep": dict(d_model=512,  n_layers=36, n_heads=8,  n_kv_heads=4,  d_ff=1536),
    "extreme_wide": dict(d_model=896,  n_layers=12, n_heads=14, n_kv_heads=7,  d_ff=2560),
    "mid_188M":   dict(d_model=768,  n_layers=28, n_heads=12, n_kv_heads=6,  d_ff=2048),
    "deep_193M":  dict(d_model=512,  n_layers=48, n_heads=8,  n_kv_heads=4,  d_ff=2048),
    "wide_207M":  dict(d_model=1024, n_layers=14, n_heads=16, n_kv_heads=8,  d_ff=3584),
}

# ─── 解码参数预设 ───────────────────────────────────────
DECODING_PRESETS = {
    "default":     dict(temperature=0.8, top_k=35, top_p=0.9),
    "creative":    dict(temperature=1.2, top_k=50, top_p=0.95),
    "conservative": dict(temperature=0.4, top_k=20, top_p=0.8),
    "deterministic": dict(temperature=0.01, top_k=1, top_p=1.0),
}

# ─── Prompt 格式处理 ────────────────────────────────────
def format_prompt(raw_prompt: str, fmt: str) -> str:
    """根据格式类型处理 prompt。"""
    if fmt == "completion":
        return raw_prompt
    elif fmt == "dialogue":
        return raw_prompt
    elif fmt == "chat":
        return raw_prompt
    else:
        return raw_prompt


def load_benchmark(benchmark_dir: Path, name: str) -> list[dict]:
    """加载 benchmark JSON 文件，返回 prompt 列表。"""
    path = benchmark_dir / f"{name}.json"
    if not path.exists():
        print(f"  ⚠ Benchmark 文件不存在: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✓ {name}: {len(data)} prompts")
    return data


def load_model(checkpoint_path: str, arch_name: str, device: torch.device):
    """加载 checkpoint 并返回 model, tokenizer, metadata。"""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    tok_path = PROJ_ROOT / "tokenizers" / "phase1_8k_real" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    arch = ARCH_CONFIGS.get(arch_name, ARCH_CONFIGS["100M"])

    cfg = ModelConfig(
        vocab_size=8192,
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        n_kv_heads=arch["n_kv_heads"],
        d_ff=arch["d_ff"],
        max_seq_len=1024,
        rope_theta=100000.0,
        dropout=0.0,
        use_flash_attention=(device.type == "cuda"),
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        use_qk_norm=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    val_ppl = ckpt.get("val_ppl", "N/A")
    steps = ckpt.get("steps", "N/A")

    return model, tok, {
        "path": str(ckpt_path),
        "name": ckpt_path.stem,
        "arch": arch_name,
        "params": param_count,
        "val_ppl": val_ppl,
        "steps": steps,
    }


def run_generation(model, tok, prompt: str, decoding: dict, device: torch.device) -> tuple[str, int]:
    """对单个 prompt 生成回答，返回 (text, token_count)。"""
    ids = tok.encode(prompt).ids
    input_ids = torch.tensor([[1] + ids], device=device)

    out_tokens = []
    try:
        for tid, is_done in model.generate_stream(
            input_ids,
            max_new_tokens=80,
            temperature=decoding["temperature"],
            top_k=decoding["top_k"],
            top_p=decoding["top_p"],
            eos_token_id=2,
        ):
            out_tokens.append(tid)
            if is_done:
                break
    except Exception as e:
        return f"[ERROR: {e}]", 0

    resp = tok.decode(out_tokens, skip_special_tokens=True)
    return resp, len(out_tokens)


def evaluate_checkpoint(model, tok, meta: dict, benchmarks: dict,
                        decoding_presets: dict, device: torch.device) -> list[dict]:
    """对单个 checkpoint 跑完所有 benchmark × 所有解码参数。"""
    results = []
    ckpt_name = meta["name"]

    for bench_name, prompts in benchmarks.items():
        if not prompts:
            continue
        fmt = bench_name

        for preset_name, decoding in decoding_presets.items():
            run_id = f"{ckpt_name}|{bench_name}|{preset_name}"
            print(f"  [{run_id}] ", end="", flush=True)

            run_results = {
                "checkpoint": meta,
                "benchmark": bench_name,
                "decoding": preset_name,
                "decoding_params": decoding,
                "generations": [],
                "error": None,
            }

            t0 = time.time()
            try:
                for item in prompts:
                    raw_prompt = item["prompt"]
                    prompt_formatted = format_prompt(raw_prompt, fmt)
                    gen_text, n_tokens = run_generation(model, tok, prompt_formatted, decoding, device)

                    run_results["generations"].append({
                        "id": item["id"],
                        "category": item.get("category", ""),
                        "prompt": raw_prompt,
                        "prompt_formatted": prompt_formatted,
                        "generation": gen_text,
                        "tokens": n_tokens,
                    })

                elapsed = time.time() - t0
                n_prompts = len(prompts)
                print(f"{n_prompts} prompts in {elapsed:.1f}s ({elapsed/n_prompts:.2f}s/prompt)")

            except Exception as e:
                elapsed = time.time() - t0
                run_results["error"] = str(e)
                print(f"ERROR after {elapsed:.1f}s: {e}")
                torch.cuda.empty_cache()

            results.append(run_results)

    return results


def main():
    parser = argparse.ArgumentParser(description="综合 Benchmark 评估")
    parser.add_argument("-c", "--checkpoints", nargs="+", required=True,
                        help="Checkpoint 文件路径")
    parser.add_argument("--arch", default="100M",
                        help="架构名称")
    parser.add_argument("--benchmarks", nargs="+", default=["completion", "dialogue", "chat"],
                        help="要跑的 benchmark: completion, dialogue, chat")
    parser.add_argument("--decoding", nargs="+", default=["default", "creative", "conservative"],
                        help="解码参数预设")
    parser.add_argument("--sweep", action="store_true",
                        help="跑全部 4 组解码参数")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 JSON 文件路径")
    parser.add_argument("--benchmark-dir", default=None,
                        help="Benchmark 文件目录")
    parser.add_argument("--device", default=None,
                        help="设备 (cuda:0 / cpu)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir)
    else:
        benchmark_dir = PROJ_ROOT / "data" / "benchmark"
    print(f"Benchmark dir: {benchmark_dir}")

    if args.sweep:
        presets = DECODING_PRESETS
    else:
        presets = {k: DECODING_PRESETS[k] for k in args.decoding if k in DECODING_PRESETS}
    print(f"Decoding presets: {list(presets.keys())}")

    print("\n─── 加载 Benchmark ───")
    benchmarks = {}
    for name in args.benchmarks:
        benchmarks[name] = load_benchmark(benchmark_dir, name)

    all_results = []
    for ckpt_path in args.checkpoints:
        print(f"\n─── 评估: {ckpt_path} ───")
        try:
            model, tok, meta = load_model(ckpt_path, args.arch, device)
            print(f"  参数: {meta['params']:,} | PPL: {meta['val_ppl']} | Steps: {meta['steps']}")

            results = evaluate_checkpoint(model, tok, meta, benchmarks, presets, device)
            all_results.extend(results)

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            continue

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJ_ROOT / "logs" / "benchmarks" / f"benchmark_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ 评估完成")
    print(f"   Checkpoints: {len(args.checkpoints)}")
    print(f"   Benchmarks:  {len(benchmarks)} × {sum(len(b) for b in benchmarks.values())} prompts")
    print(f"   Decodings:   {len(presets)}")
    print(f"   总运行次数:  {len(all_results)}")
    print(f"   结果保存到:  {output_path}")

    print(f"\n─── 快速摘要 ───")
    for run in all_results:
        ckpt_name = run["checkpoint"]["name"]
        bench = run["benchmark"]
        dec = run["decoding"]
        gens = run["generations"]
        avg_tokens = sum(g["tokens"] for g in gens) / max(len(gens), 1)
        first_gen = gens[0]["generation"][:80] if gens else "N/A"
        print(f"  [{ckpt_name}] {bench}/{dec}: avg_tokens={avg_tokens:.0f}")
        print(f"    示例: {gens[0]['prompt'][:30]} → {first_gen}...")


if __name__ == "__main__":
    main()
