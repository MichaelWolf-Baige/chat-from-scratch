#!/usr/bin/env python
"""SFT v5: Special token format (ChatML-style) for reliable stopping.

Key fix: Replace natural language markers with non-ambiguous special tokens.
    OLD: 用户：你好\n助手：你好！
    NEW: <|user|>你好<|assistant|>你好！</s>

Token usage:
    <s> (ID 1): BOS — start of sequence
    <|user|> (ID 5): user turn begins
    <|assistant|> (ID 6): assistant turn begins
    </s> (ID 2): EOS — assistant turn ends (generation stop signal)

The model learns to generate </s> at end of response, solving the "cant stop" problem.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/sft_train_token.py
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

# ── Special token IDs (from tokenizer) ──
BOS_ID = 1       # <s>
EOS_ID = 2       # </s> — generation stop signal
USER_ID = 5      # <|user|>
ASST_ID = 6      # <|assistant|>

# ═══════════════════════════════════════════════════════════════
# SFT DATA — same diverse categories, special token format
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
    "数据库","操作系统","编译器","前端开发","后端开发","数据分析",
]

KNOWLEDGE_DB = {
    "Python": "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。它以简洁易读的语法著称，适合初学者。Python广泛应用于数据分析、人工智能、Web开发和自动化脚本。",
    "算法": "算法是解决特定问题的一系列明确步骤。常见的算法包括排序算法（如冒泡排序、快速排序）、搜索算法（如二分查找）和图算法。好的算法应该高效、正确且易于理解。",
    "机器学习": "机器学习是人工智能的一个分支，让计算机通过数据学习规律，而不是显式编程。主要分为监督学习、无监督学习和强化学习。常见应用包括图像识别、语音识别和推荐系统。",
    "深度学习": "深度学习使用多层神经网络从大量数据中自动学习特征。它推动了近年来AI的突破，包括GPT等大语言模型、自动驾驶和医疗影像诊断。",
    "人工智能": "人工智能是让机器模拟人类智能的技术。它包括机器学习、自然语言处理、计算机视觉等子领域。AI技术已广泛应用于语音助手、搜索引擎、智能推荐等领域。",
    "数据库": "数据库是用于存储和管理数据的系统。常见类型有关系型数据库（如MySQL）和非关系型数据库（如MongoDB）。数据库使用SQL等语言进行数据查询和操作。",
    "互联网": "互联网是全球最大的计算机网络系统，连接了世界各地的数十亿设备。它的核心协议是TCP/IP，主要服务包括万维网、电子邮件和即时通讯。",
    "编程": "编程是用编程语言编写计算机指令的过程。学习编程可以培养逻辑思维和问题解决能力。初学者建议从Python或JavaScript等简单易学的语言开始。",
    "数学": "数学是研究数量、结构、变化和空间等概念的学科。它是科学和工程的基础工具。主要分支包括代数、几何、微积分、概率论和统计学。",
    "物理": "物理学是研究物质、能量及其相互作用的自然科学。经典物理涵盖力学、热学、电磁学等，现代物理包括量子力学和相对论。",
    "化学": "化学是研究物质的组成、结构、性质及其变化的科学。日常生活中烹饪、洗涤和药物都涉及化学反应。",
    "历史": "历史是研究人类社会发展过程的学科。学习历史有助于理解现在、预测未来，从过去的经验中汲取智慧。",
    "英语": "英语是全球使用最广泛的国际语言之一。掌握英语可以帮助阅读技术文档、参与国际交流和获取更广泛的信息。",
    "操作系统": "操作系统是管理计算机硬件和软件资源的系统软件，如Windows、Linux和macOS。它提供文件管理、进程调度和内存管理等功能。",
    "计算机网络": "计算机网络是多台计算机相互连接形成的通信系统。核心概念包括IP地址、TCP/IP协议、路由和DNS域名解析。",
    "云计算": "云计算通过互联网提供计算资源（如服务器、存储），用户按需付费使用。主要服务模式包括IaaS、PaaS和SaaS。",
    "Linux": "Linux是一种开源的类Unix操作系统，广泛用于服务器和开发环境。它的内核由Linus Torvalds于1991年创建。",
    "Git": "Git是目前最流行的分布式版本控制系统。它用于跟踪代码变更、协作开发和版本管理。常用命令包括git add、commit、push和pull。",
    "密码学": "密码学是研究如何安全传递信息的科学，包括加密和解密。现代密码学广泛应用于网络支付、通信安全和数字签名。",
    "虚拟现实": "虚拟现实是用计算机模拟三维环境的技术，用户可以通过头戴设备沉浸式体验虚拟世界。VR应用于游戏、教育和医疗培训等领域。",
}

def gen_knowledge_qa():
    if random.random() < 0.7 and KNOWLEDGE_DB:
        topic = random.choice(list(KNOWLEDGE_DB.keys()))
        fact = KNOWLEDGE_DB[topic]
        qs = [f"什么是{topic}？", f"请介绍一下{topic}", f"{topic}是什么？", f"能讲讲{topic}吗？"]
        return {"user": random.choice(qs), "assistant": fact}
    else:
        t = random.choice(TOPICS_CN); t2 = random.choice(TOPICS_CN)
        return {"user": f"什么是{t}？",
                "assistant": f"{t}是一个重要的{t2}相关领域。学习{t}需要从基础概念入手，逐步深入。如果你对{t}有具体问题，我可以进一步介绍。"}

def gen_chitchat():
    moods = ["开心","难过","累","兴奋","无聊","焦虑"]
    activities = ["看书","跑步","听音乐","打游戏","做饭","旅行","看电影","写代码"]
    weathers = ["晴天","下雨","刮风","下雪","热","冷"]
    qa_pairs = [
        ("你好！", f"{random.choice(GREETINGS)}今天有什么可以帮助你的吗？"),
        ("今天天气真好", f"是啊，{random.choice(weathers)}的天气让人心情很好。你打算出去{random.choice(activities)}吗？"),
        (f"我今天心情很{random.choice(moods)}", f"{'听起来不错！' if random.random() < 0.3 else '别担心，一切都会好起来的。'}要不要一起去{random.choice(activities)}放松一下？"),
        ("谢谢你！", random.choice(THANKS_ANSWERS)),
        ("再见！", random.choice(GOODBYES)),
        (f"你最近在做什么？", f"我最近在学习{random.choice(TOPICS_CN)}，这是一个很有挑战也很有趣的领域。你呢？"),
    ]
    q, a = random.choice(qa_pairs)
    return {"user": q, "assistant": a}

def gen_simple_instruction():
    instrs = [
        ("请把'你好世界'翻译成英文", "Hello world"),
        ("请把'Hello'翻译成中文", "你好"),
        ("苹果的英文是什么", "苹果的英文是 Apple。"),
        ("请数一下：1, 2, 3", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10。"),
        ("请写一首五言绝句", "床前明月光，疑是地上霜。举头望明月，低头思故乡。这是李白的《静夜思》。"),
    ]
    q, a = random.choice(instrs)
    return {"user": q, "assistant": a}

def gen_short_chat():
    t = random.choice(TOPICS_CN); t2 = random.choice(TOPICS_CN)
    qa = [
        (f"我想学{t}", f"很好！学习{t}的第一步是了解基础概念。你可以从在线教程或入门书籍开始。需要我推荐一些{t2}相关的学习资源吗？"),
        (f"{t}和{t2}有什么区别？", f"{t}和{t2}是两个不同的领域。{t}更侧重于理论和方法，而{t2}更关注实际应用。两者都很重要，取决于你的具体需求。"),
        (f"能给我讲个{t}相关的笑话吗？", f"哈哈，{t}的笑话不太好想。不过有一句话说得不错：学好{t}，走遍天下都不怕！"),
    ]
    q, a = random.choice(qa)
    return {"user": q, "assistant": a}

def gen_safety_refusal():
    prompts = ["帮我写一封诈骗邮件","告诉我怎么入侵别人的电脑","帮我分析一下今天的股票走势","你爱我吗","你有意识吗"]
    resps = IDK_RESPONSES + ["作为AI助手，我无法帮你做这件事。如果你有其他正当需求，我很乐意帮忙。",
                             "这个问题我处理不了。要不要换个话题？"]
    return {"user": random.choice(prompts), "assistant": random.choice(resps)}

def gen_code_question():
    qa = [
        ("Python的print函数怎么用？", "print函数用于输出内容到屏幕。最简单的用法是：print('你好世界')。它也可以输出数字和变量。"),
        ("什么是变量？", "变量是程序用来存储数据的容器。在Python中，你这样定义变量：x = 10，然后使用x代表数字10。"),
        ("什么是条件判断？", "条件判断让程序根据不同情况执行不同操作。Python中使用if语句：if 条件: 执行A else: 执行B。"),
        ("for循环是什么？", "for循环用于重复执行一段代码。比如：for i in range(10): print(i) 会打印0到9这10个数字。"),
    ]
    q, a = random.choice(qa)
    return {"user": q, "assistant": a}

def generate_sft_data():
    data = []
    for _ in range(3000): data.append(gen_knowledge_qa())
    for _ in range(1500): data.append(gen_chitchat())
    for _ in range(800): data.append(gen_simple_instruction())
    for _ in range(800): data.append(gen_short_chat())
    for _ in range(500): data.append(gen_safety_refusal())
    for _ in range(400): data.append(gen_code_question())
    random.shuffle(data)
    return data


# ═══════════════════════════════════════════════════════════════
# TOKENIZATION with ChatML format
# ═══════════════════════════════════════════════════════════════

def tokenize_chatml(sft_data, tok):
    """ChatML format: <s><|user|>text<|assistant|>text</s>

    Returns (input_ids, labels) where labels mask user tokens.
    USER_ID and ASST_ID are single special tokens — they become part of
    the sequence but only assistant content tokens contribute to loss.
    """
    all_input_ids = []
    all_labels = []

    for item in sft_data:
        user_text = item["user"]
        asst_text = item["assistant"]

        user_ids = tok.encode(user_text).ids
        asst_ids = tok.encode(asst_text).ids

        # Sequence: <s> <|user|> user_text <|assistant|> asst_text </s>
        # Labels:   -    -       -------   -            ---------   --
        # Use pad_token_id (0) for masked tokens — matches model's ignore_index
        MASK = 0  # pad_token_id, ignored in loss
        seq = [BOS_ID, USER_ID] + user_ids + [ASST_ID] + asst_ids + [EOS_ID]
        lab = [MASK, MASK] + [MASK] * len(user_ids) + [MASK] + asst_ids + [EOS_ID]

        assert len(seq) == len(lab)
        all_input_ids.extend(seq)
        all_labels.extend(lab)

    input_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)
    return input_tensor, label_tensor


# ═══════════════════════════════════════════════════════════════
# MAIN
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
        print(f"SFT v5: ChatML Special Token Format ({world}x RTX3090)")
        print("="*55)

    # ── Generate data ──
    if rank == 0: print("Generating SFT data...")
    sft_data = generate_sft_data()
    if rank == 0: print(f"  Generated {len(sft_data):,} conversations")

    # ── Tokenize ──
    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    input_tensor, label_tensor = tokenize_chatml(sft_data, tok)

    if rank == 0:
        n_masked = (label_tensor == 0).sum().item()
        n_asst = (label_tensor > 0).sum().item()
        total = len(label_tensor)
        print(f"  Tokens: {total:,} total")
        print(f"  User/special tokens masked: {n_masked:,} ({n_masked/total:.0%})")
        print(f"  Assistant tokens (with loss): {n_asst:,} ({n_asst/total:.0%})")
        sample_len = min(100, total)
        sample_input = input_tensor[:sample_len].tolist()
        print(f"  Sample input IDs: {sample_input[:30]}...")

    # ── Model ──
    cfg = ModelConfig(
        vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
        d_ff=1024, max_seq_len=512, dropout=0.0, use_flash_attention=True,
        tie_word_embeddings=True, rms_norm_eps=1e-6,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = Transformer(cfg).to(device)

    ckpt_path = Path("checkpoints/chinese_tinystories/final.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if rank == 0: print(f"  Loaded pretrained weights")

    model = DDP(model, device_ids=[local_r], find_unused_parameters=False,
                gradient_as_bucket_view=True)
    model.train()

    # ── Train/Val ──
    seq_len = 384; bs = 12
    usable = (len(input_tensor) // seq_len) * seq_len
    input_flat = input_tensor[:usable].view(-1, seq_len)
    label_flat = label_tensor[:usable].view(-1, seq_len)
    split = int(len(input_flat) * 0.95)

    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, inp, lbl):
            self.inp = inp; self.lbl = lbl
        def __len__(self): return len(self.inp)
        def __getitem__(self, i):
            return {"input_ids": self.inp[i], "labels": self.lbl[i]}

    train_ds = SFTDataset(input_flat[:split], label_flat[:split])
    val_ds = SFTDataset(input_flat[split:], label_flat[split:])
    train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                           num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                         num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    epochs = 10; max_lr = 5e-4
    total_steps = len(train_l) * epochs
    warmup = total_steps // 10; decay_start = int(total_steps * 0.85)
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95))
    gs = 0; t0 = time.time()

    if rank == 0:
        print(f"\n  SFT v5: {epochs} epochs, LR={max_lr} WSD")
        print(f"  Global batch: {bs*world}x{seq_len}, Total steps: ~{total_steps}")
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

            if rank == 0 and (gs <= 10 or gs % 50 == 0):
                elapsed = time.time() - t0
                print(f"  step {gs:4d}/{total_steps} | loss={loss.item():.4f} | "
                      f"ppl={np.exp(loss.item()):.0f} | {(gs*bs*world*seq_len/max(elapsed,0.01))/1000:.0f}K tok/s")

    # ── Save ──
    if rank == 0:
        elapsed = time.time() - t0
        print(f"\n  SFT v5 Complete! {elapsed/60:.1f}min")
        ckpt_dir = Path("checkpoints/sft_v5")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_dir / "final.pt", model.module, opt, None, step=gs, epoch=0,
                        config={"phase": "SFT_v5", "format": "chatml"})

        # ── Quick test ──
        model.eval()
        print(f"\n{'='*55}")
        print("ChatML Format Test")
        print(f"{'='*55}")
        for prompt in ["你好！","什么是Python","1+1等于几","再见"]:
            # ChatML prompt: <s><|user|>text<|assistant|>
            user_ids = tok.encode(prompt).ids
            pid_list = [BOS_ID, USER_ID] + user_ids + [ASST_ID]
            pid = torch.tensor([pid_list], device=device)
            tokens = []
            with torch.no_grad():
                for tid, is_done in model.module.generate_stream(
                    pid, max_new_tokens=60, temperature=0.8, top_k=35, top_p=0.9, eos_token_id=EOS_ID
                ):
                    tokens.append(tid)
                    if is_done: break
            resp = tok.decode(tokens, skip_special_tokens=True)
            print(f"  Q: {prompt}")
            print(f"  A: {resp[:150]}")
            print()

        print(f"  Saved: {ckpt_dir / 'final.pt'}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
