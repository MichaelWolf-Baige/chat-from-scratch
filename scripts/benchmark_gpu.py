# GPU benchmark: forward/backward throughput, generation speed
import sys
sys.path.insert(0, ".")
import torch
import time
from src.model.config import ModelConfig
from src.model.transformer import Transformer

device = torch.device("cuda:0")
torch.cuda.reset_peak_memory_stats()

# ── Model ──
cfg = ModelConfig.phase1()
print(f"Model: {cfg.total_params:,} params")
model = Transformer(cfg).to(device)
model.train()

# ── Data ──
B, S = 32, 512
input_ids = torch.randint(1, 8192, (B, S), device=device)
labels = input_ids.clone()

# Warmup
for _ in range(3):
    _ = model(input_ids)
torch.cuda.synchronize()

# ── Forward + Backward timing ──
N = 10
times = []
for _ in range(N):
    torch.cuda.synchronize()
    t0 = time.time()
    _, outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    torch.cuda.synchronize()
    times.append(time.time() - t0)

avg_time = sum(times) / len(times)
tokens_per_step = B * S
tokens_per_sec = tokens_per_step / avg_time

print(f"\nForward+Backward ({N} steps):")
print(f"  Batch: {B}x{S} = {tokens_per_step:,} tokens/step")
print(f"  Avg time: {avg_time*1000:.1f} ms/step")
print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
print(f"  Memory: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB peak")

# ── Generation benchmark ──
model.eval()
prompt = torch.randint(1, 1000, (1, 20), device=device)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    full, new = model.generate(prompt, max_new_tokens=50, temperature=0.8)
torch.cuda.synchronize()
gen_time = time.time() - t0

print(f"\nGeneration:")
print(f"  Generated 50 tokens in {gen_time*1000:.0f} ms")
print(f"  Speed: {50/gen_time:.0f} tokens/sec")

# ── Scale estimation ──
print(f"\n── Scale Estimates ──")
print(f"Phase 1 (14M, 1B tokens, 3090):")
print(f"  Steps: {1_000_000_000 / (32*512):,.0f} (bs=32, seq=512)")
print(f"  Estimated time: {1_000_000_000 / (32*512) * avg_time / 3600:.1f} hours")

# Larger config test
print(f"\nPhase 2 (49M, bs=64, seq=2048) estimate:")
cfg2 = ModelConfig.phase2()
model2 = Transformer(cfg2).to(device)
B2, S2 = 16, 1024  # smaller batch to fit memory
input_ids2 = torch.randint(1, cfg2.vocab_size, (B2, S2), device=device)
labels2 = input_ids2.clone()
model2.train()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

for _ in range(3):
    _ = model2(input_ids2)
torch.cuda.synchronize()

times2 = []
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.time()
    _, out2 = model2(input_ids2, labels=labels2)
    out2["loss"].backward()
    torch.cuda.synchronize()
    times2.append(time.time() - t0)

avg2 = sum(times2) / len(times2)
tps2 = (B2 * S2) / avg2
mem2 = torch.cuda.max_memory_allocated() / 1024**3
print(f"  {cfg2.total_params:,} params, bs={B2}, seq={S2}")
print(f"  {times2[0]*1000:.0f}ms/step, {tps2:,.0f} tok/s, {mem2:.1f} GB VRAM")
print(f"  Phase 2 10B tokens: ~{10_000_000_000 / tps2 / 3600:.0f} hours")

del model2, input_ids2, labels2
torch.cuda.empty_cache()

print(f"\n✅ All GPU benchmarks complete!")
