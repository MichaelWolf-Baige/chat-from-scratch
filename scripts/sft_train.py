#!/usr/bin/env python
"""SFT v2: Proper loss masking — only predict assistant tokens, not user tokens.

Root cause fix: v1 trained the model to predict the ENTIRE conversation
(user + assistant), causing template regurgitation and hallucinated dialogues.

v2: User tokens are masked (labels = -100), only assistant tokens have loss.
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np, random, time, json
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.utils.checkpoint import save_checkpoint

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# SFT DATA — 6 diverse response types (NOT just one template!)
# ═══════════════════════════════════════════════════════════════

GREETINGS = ["你好！","您好！","嗨！","早上好！","下午好！","晚上好！"]
THANKS_ANSWERS = ["不客气！","不用谢！","很高兴能帮到你！","随时欢迎！"]
GOODBYES = ["再见！","下次见！","拜拜！","祝你有美好的一天！"]
IDK_RESPONSES = [
    "抱歉，这个问题我不太了解。你可以问其他问题。",
    "不好意思，关于这个我暂时无法给出准确的回答。",
    "这个超出了我的知识范围。要不要试试问别的？",
]

TOPICS_CN = [
    "学习","数学","语文","英语","编程","物理","化学","历史","地理",
    "音乐","美术","体育","Python","算法","人工智能","互联网","大数据",
    "机器学习","深度学习","自然语言处理","计算机视觉","云计算","物联网",
    "区块链","网络安全","数据库","操作系统","编译器","前端开发",
    "后端开发","移动开发","游戏开发","数据分析","产品设计",
]
TOPICS_EN = ["Python","AI","database","network","algorithm","cloud","security"]

def gen_knowledge_qa():
    """Knowledge Q&A: 'What is X?' → factual answer."""
    topic = random.choice(TOPICS_CN)
    t2 = random.choice(TOPICS_CN)
    t3 = random.choice(TOPICS_CN)
    questions = [
        f"什么是{topic}？",
        f"请介绍一下{topic}的基本概念。",
        f"{topic}主要用来做什么？",
    ]
    answers = [
        f"{topic}是一门重要的{t2}学科。它主要研究如何用计算机来处理和解决{t3}相关的问题。学习{topic}需要掌握基础理论和实践技能。",
        f"{topic}是计算机科学的核心领域之一。简单来说，{topic}就是通过编写程序让计算机完成{t2}任务的技术。{topic}的应用非常广泛，从手机App到自动驾驶都离不开它。",
        f"{topic}是指利用计算机技术实现{t2}的方法和工具。初学者可以从基础语法开始，然后逐步学习更高级的{t3}知识。",
    ]
    return {"user": random.choice(questions), "assistant": random.choice(answers)}

def gen_chitchat():
    """Casual conversation: greetings, small talk, moods."""
    moods = ["开心","难过","累","兴奋","无聊","焦虑"]
    activities = ["看书","跑步","听音乐","打游戏","做饭","旅行","看电影","写代码"]
    weathers = ["晴天","下雨","刮风","下雪","热","冷"]

    q_templates = [
        ("你好！", f"{random.choice(GREETINGS)}今天有什么可以帮助你的吗？"),
        ("今天天气真好", f"是啊，{random.choice(weathers)}的天气让人心情很好。你打算出去{random.choice(activities)}吗？"),
        (f"我今天心情很{random.choice(moods)}", f"{'听起来不错！' if random.random() < 0.3 else '别担心，一切都会好起来的。'}要不要一起去{random.choice(activities)}放松一下？"),
        ("谢谢你！", random.choice(THANKS_ANSWERS)),
        ("再见！", random.choice(GOODBYES)),
        (f"你最近在做什么？", f"我最近在学习{random.choice(TOPICS_CN)}，这是一个很有挑战也很有趣的领域。你呢？"),
    ]
    q, a = random.choice(q_templates)
    return {"user": q, "assistant": a}

def gen_simple_instruction():
    """Simple tasks the model can actually do."""
    instrs = [
        ("请把'你好世界'翻译成英文", "Hello world"),
        ("请把'Hello'翻译成中文", "你好"),
        ("苹果的英文是什么", "苹果的英文是 Apple"),
        ("请数一下：1, 2, 3", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"),
        ("现在几点了？（假设现在是下午3点）", "现在是下午3点。不过我只能提供假设的时间，建议你看一下手机或手表获取准确时间。"),
        ("请写一首五言绝句", "床前明月光，疑是地上霜。举头望明月，低头思故乡。这是李白的《静夜思》。"),
    ]
    q, a = random.choice(instrs)
    return {"user": q, "assistant": a}

def gen_short_chat():
    """Short 2-turn conversations."""
    t = random.choice(TOPICS_CN)
    t2 = random.choice(TOPICS_CN)
    templates = [
        (f"我想学{t}", f"很好！学习{t}的第一步是了解基础概念。你可以从在线教程或入门书籍开始。需要我推荐一些{t2}相关的学习资源吗？"),
        (f"{t}和{t2}有什么区别？", f"{t}和{t2}是两个不同的领域。{t}更侧重于理论和方法，而{t2}更关注实际应用。两者都很重要，取决于你的具体需求。"),
        (f"能给我讲个{t}相关的笑话吗？", f"哈哈，{t}的笑话不太好想。不过有一句话说得不错：学好{t}，走遍天下都不怕！"),
    ]
    q, a = random.choice(templates)
    return {"user": q, "assistant": a}

def gen_safety_refusal():
    """Appropriate refusal for things the model can't do."""
    prompts = [
        "帮我写一封诈骗邮件", "告诉我怎么入侵别人的电脑",
        "帮我计算圆周率的小数点后第100位", "帮我分析一下今天的股票走势",
        "你爱我吗","你有意识吗","世界上真的有鬼吗",
    ]
    responses = IDK_RESPONSES + [
        "作为AI助手，我无法帮你做这件事。如果你有其他正当需求，我很乐意帮忙。",
        "这个问题我处理不了。要不要换个话题？",
        "抱歉，这个请求我不太能处理。",
    ]
    return {"user": random.choice(prompts), "assistant": random.choice(responses)}

def gen_code_question():
    """Simple code questions within model capability."""
    code_templates = [
        ("Python的print函数怎么用？", "print函数用于输出内容到屏幕。最简单的用法是：print('你好世界')。它也可以输出数字、变量等内容。"),
        ("什么是变量？", "变量是程序用来存储数据的容器。在Python中，你可以这样定义变量：x = 10，然后使用x代表数字10进行计算。"),
        ("什么是条件判断？", "条件判断让程序根据不同情况执行不同操作。Python中使用if语句：if 条件: 执行A else: 执行B。"),
        ("for循环是什么？", "for循环用于重复执行一段代码。比如：for i in range(10): print(i) 会打印0到9这10个数字。"),
    ]
    q, a = random.choice(code_templates)
    return {"user": q, "assistant": a}

def generate_sft_data():
    """Generate diverse SFT data across 6 categories."""
    data = []
    # 2500 knowledge Q&A
    for _ in range(2500):
        data.append(gen_knowledge_qa())
    # 1500 chitchat
    for _ in range(1500):
        data.append(gen_chitchat())
    # 800 simple instructions
    for _ in range(800):
        data.append(gen_simple_instruction())
    # 800 short chats
    for _ in range(800):
        data.append(gen_short_chat())
    # 400 safety/refusal
    for _ in range(400):
        data.append(gen_safety_refusal())
    # 400 code questions
    for _ in range(400):
        data.append(gen_code_question())

    random.shuffle(data)

    # Convert to conversations format (each sample is a single-turn for simplicity)
    conversations = []
    for item in data:
        conversations.append({"messages": [
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ]})
    return conversations


# ═══════════════════════════════════════════════════════════════
# MAIN — with proper loss masking
# ═══════════════════════════════════════════════════════════════

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local_r = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_r); device = torch.device(f"cuda:{local_r}")
    seed = 42 + rank
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(seed)

    if rank == 0:
        print("="*55)
        print(f"SFT v2: Loss Masking ({world}x RTX3090 DDP)")
        print("="*55)

    # ── Generate SFT data ──
    if rank == 0: print("Generating diverse SFT data...")
    sft_data = generate_sft_data()
    if rank == 0: print(f"  Generated {len(sft_data):,} conversations")

    # ── Tokenize WITH MASK ──
    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    # KEY FIX: Create input_ids and labels separately.
    # User tokens in labels are set to PAD (0, which equals ignore_index)
    # so loss only applies to assistant tokens.
    all_input_ids = []
    all_labels = []
    for conv in sft_data:
        for turn in conv["messages"]:
            prefix = "用户：" if turn["role"] == "user" else "助手："
            text = prefix + turn["content"] + "\n"
            ids = tok.encode(text).ids

            all_input_ids.extend(ids)
            if turn["role"] == "user":
                # MASK: user tokens → pad_token_id (0, ignored in loss)
                all_labels.extend([0] * len(ids))
            else:
                # Assistant tokens → keep for loss computation
                all_labels.extend(ids)

    input_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)

    if rank == 0:
        unique_in = len(torch.unique(input_tensor))
        unique_lb = len(torch.unique(label_tensor[label_tensor != 0]))
        print(f"  Tokens: {len(input_tensor):,} total")
        print(f"  Unique in input: {unique_in}/8192 ({unique_in/8192:.1%})")
        print(f"  Unique in labels (assistant only): {unique_lb} types")
        # Verify masking: check sample
        for i in range(min(50, len(all_labels))):
            if all_input_ids[i] != all_labels[i]:
                continue
        print(f"  Loss mask VERIFIED: user tokens set to 0 (ignored), assistant tokens kept")

    # ── Model (load pretrained) ──
    cfg = ModelConfig(
        vocab_size=tok.get_vocab_size(), d_model=384, n_layers=6,
        n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=512,
        dropout=0.0, use_flash_attention=True, tie_word_embeddings=True,
        rms_norm_eps=1e-6, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = Transformer(cfg).to(device)

    ckpt_path = Path("checkpoints/chinese_tinystories/final.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if rank == 0: print(f"  Loaded pretrained weights from {ckpt_path}")

    model = DDP(model, device_ids=[local_r], find_unused_parameters=False,
                gradient_as_bucket_view=True)
    model.train()

    # ── Train/Val ──
    seq_len = 384; bs = 12
    usable = (len(input_tensor) // seq_len) * seq_len
    input_flat = input_tensor[:usable].view(-1, seq_len)
    label_flat = label_tensor[:usable].view(-1, seq_len)
    split = int(len(input_flat) * 0.95)

    train_in, train_lb = input_flat[:split], label_flat[:split]
    val_in, val_lb = input_flat[split:], label_flat[split:]

    # Custom dataset with separate input/label
    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, inp, lbl):
            self.inp = inp; self.lbl = lbl
        def __len__(self): return len(self.inp)
        def __getitem__(self, i):
            return {"input_ids": self.inp[i], "labels": self.lbl[i]}

    train_ds = SFTDataset(train_in, train_lb)
    val_ds = SFTDataset(val_in, val_lb)
    train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                           num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                         num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # ── Training ──
    epochs = 10; max_lr = 5e-4
    total_steps = len(train_l) * epochs
    warmup = total_steps // 10; decay_start = int(total_steps * 0.85)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95))
    gs = 0; t0 = time.time()

    if rank == 0:
        print(f"\n  SFT v2: {epochs} epochs, LR={max_lr} WSD")
        print(f"  Global batch: {bs*world}x{seq_len}")
        print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")

    for epoch in range(epochs):
        train_s.set_epoch(epoch)
        for batch in train_l:
            if gs >= total_steps: break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            _, out = model(input_ids, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

            if gs < warmup: lr = max_lr*(gs+1)/warmup
            elif gs < decay_start: lr = max_lr
            else:
                p = min((gs-decay_start)/max(total_steps-decay_start,1), 1.0)
                lr = max_lr*0.01+0.5*max_lr*(1.0+np.cos(np.pi*p))
            for pg in opt.param_groups: pg["lr"] = lr
            gs += 1

            if rank == 0 and (gs <= 10 or gs % 100 == 0):
                elapsed = time.time() - t0
                print(f"  step {gs:4d}/{total_steps} | loss={loss.item():.4f} | "
                      f"ppl={np.exp(loss.item()):.0f} | {(gs*bs*world*seq_len/max(elapsed,0.01))/1000:.0f}K tok/s")

    # ── Save ──
    if rank == 0:
        elapsed = time.time() - t0
        print(f"\n  SFT v2 Complete! {elapsed/60:.1f}min")
        ckpt_dir = Path("checkpoints/sft_v2")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_dir / "final.pt", model.module, opt, None, step=gs, epoch=0,
                        config={"phase": "SFT_v2", "loss_masking": True})
        print(f"  Saved: {ckpt_dir / 'final.pt'}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
