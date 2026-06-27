# Phase 1 完整问题日志

> 日期：2026-06-25 ~ 2026-06-26
> 项目：chat-from-scratch
> 目标：从零训练 14M Llama-style Transformer，验证全管线

---

## 问题分类体系

| 标签 | 含义 |
|------|------|
| 🔴 阻塞性 | 不修复无法继续 |
| 🟡 性能/效率 | 不阻塞但影响速度 |
| 🟢 学习价值 | 不阻塞，但理解它能避免未来踩坑 |

---

## 一、项目搭建阶段（2026-06-25）

### 问题 1：RoPE cos/sin 缓存维度广播失败 {#P1}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性 |
| **发现阶段** | 38 项测试运行 |

**症状**：`RuntimeError: The size of tensor a (6) must match the size of tensor b (128) at non-singleton dimension 3`

**原因**：`RotaryEmbedding.__init__` 存储的 cos/sin 形状是 `(seq_len, d_head//2)`——只存了一半频率。前向时直接 unsqueeze 得到 `(1, S, 1, 32)`，但 attention head 的 `d_head=64`，`apply_rotary_emb` 里对 x 做 `x[..., 0::2]` 后是 `(B, S, H, 32)`，维度对不上。

具体错误链：
```python
# 错误的缓存
angles = torch.outer(positions, freqs)  # (S, d_head//2)
self.cos_cached = angles.cos()          # (S, 32)，丢了从 32→64 的配对信息

# forward 中
cos = cos.unsqueeze(0).unsqueeze(2)     # (1, S, 1, 32) —— 只有 32 维
x_even = x[..., 0::2]                   # (B, S, H, 32)  —— 对不上
```

**解决方案**：用 `repeat_interleave(2, dim=-1)` 在初始化时把 `(S, d_head//2)` 展开到 `(S, d_head)`。这样缓存和使用时维度天然匹配。

```python
# 修正
angles = torch.outer(positions, freqs)           # (S, d_head//2)
angles_full = angles.repeat_interleave(2, dim=-1) # (S, d_head)
self.cos_cached = angles_full.cos()
# forward 中 cos 是 (B, S, 1, 64)，x 也是 (B, S, H, 64)，broadcast 正确
```

**配套修正**：`apply_rotary_emb` 中也需要对 cos/sin 做 `[..., 0::2]` 切片以匹配 x 的半维度。

**耗时**：5 分钟定位 + 10 分钟修复

**如果再遇到怎么更快定位**：先打印 `cos.shape`、`x.shape` 和 `x[..., 0::2].shape`，三者维度关系一目了然。

---

### 问题 2：循环导入导致模块无法加载 {#P2}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性（Python 包管理） |
| **发现阶段** | 运行测试 `tests/test_data.py` |

**症状**：`ImportError: cannot import name 'PretrainDataset' from partially initialized module 'src.data.dataset'`

**原因**：`src/data/dataset.py` 的内容只有一行：
```python
from src.data.dataset import PretrainDataset, ...  # 自己导入自己！
```
而实际的类定义也在 `__init__.py` 里，形成了「`__init__` → `dataset.py` → `__init__`」的循环。

**解决方案**：把所有的类定义从 `__init__.py` 移到 `dataset.py`，`__init__.py` 只做 re-export。

```python
# src/data/__init__.py (修正后)
from src.data.dataset import PretrainDataset, PretrainIterableDataset, make_dataloader
```

**经验教训**：`__init__.py` 只应该导入和暴露接口，永远不要在子模块里导入 `__init__` 或自己。实际的类/函数放在命名文件中。

**耗时**：2 分钟

---

### 问题 3：Phase 2 模型参数预算超出预期 {#P3}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 非阻塞（测试失败） |
| **问题类型** | 概念性（参数计算） |
| **发现阶段** | 参数验证测试 |

**症状**：`test_phase2_params` 断言失败：实际 **69,975,680** 参数，预期 40M-60M。

**原因**：配置 `vocab_size=16384, d_model=640, n_layers=12, n_heads=10, d_ff=1728` 算出来约 70M。SwiGLU FFN 的 `d_ff` 取大了，Embedding 层 16K×640=10.5M 也比预期的多。

**解决方案**：调整 Phase 2 规格为 `d_model=576, n_layers=10, n_heads=9, d_ff=1536`，约 49M。

**经验教训**：配置写完必须手算参数预算。SwiGLU 是 `3*d_model*d_ff` 不是 `2*d_model*d_ff`。Embedding 层在小模型里占比巨大（10M/70M=14%），要专门关注。

**耗时**：5 分钟

---

### 问题 4：GitHub 提交者显示错误身份 {#P4}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 非阻塞 |
| **问题类型** | 工程性（Git 配置） |
| **发现阶段** | 推送后查看 GitHub |

**症状**：Commits 显示 author 为 `86136`（Windows 本地用户名），Contributors 中出现幽灵账户。

**原因**：
1. `git config user.name` 被设为 `"86136"`（`C:\Users\86136` 的目录名，不是 GitHub 身份）
2. `git config user.email` 被设为编造的 `86136@users.noreply.github.com`
3. Commit message 中的 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` 创建了第二个幽灵贡献者

**解决方案**：
1. 修正 `git config` 为 `MichaelWolf-Baige / baige126@qq.com`
2. 去掉 commit message 中的 `Co-Authored-By` 行
3. 删除旧仓库 → 新建 → push 干净历史（因为旧提交的哈希已缓存在 GitHub 对象库，无法通过 push force 清除）

**经验教训**：
- 初始化项目时第一件事就是设置 git config
- GitHub 看的是 **email**（不是 username）来匹配提交者
- `Co-Authored-By` 只有在 email 对应真实 GitHub 账户时才有效

**耗时**：30 分钟（包括 research + 删库重建）

---

## 二、冒烟测试阶段（2026-06-26）

### 问题 5：脚本缺少 PYTHONPATH，无法导入 src 模块 {#P5}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性（Python 路径） |
| **发现阶段** | 服务器运行 `preprocess_data.py` |

**症状**：`ModuleNotFoundError: No module named 'src'`

**原因**：`scripts/preprocess_data.py` 没有 `sys.path` 插入项目根目录。本地因为编辑器/Python 配置了路径所以能跑，服务器上暴露了。

**解决方案**：在脚本顶部加：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**最佳实践**：项目的**所有可执行脚本**都应该在最顶部做这件事，不依赖环境变量或 IDE 配置。

**耗时**：2 分钟

---

### 问题 6：合成数据 tokenizer 中文 100% UNK {#P6}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟢 预期行为（不是 bug） |
| **问题类型** | 概念性（数据规模） |
| **发现阶段** | 合成数据冒烟测试 |

**症状**：10K 条合成文本（50% 中文）、词汇量 473（目标 4096），中文文本 100% 变为 `<unk>`。

**诊断**：数据量太小（10K 行 × ~38 字 = 380K 字符），无法为 8192 目标词表学到有意义的合并。英文有 byte-level fallback（ByteLevel pre-tokenizer），中文没有类似机制。

**验证方法**：确认这是**数据量问题**而非代码 bug——正式数据（146K 文档、436MB）训练后词汇量 8192、中文 0.64 char/token、UNK 率 0%，问题自动消失。

**经验教训**：冒烟测试中预期行为 ≠ 生产行为。不需要为合成数据的表现 panic。

**耗时**：5 分钟（确认不阻塞后继续）

---

### 问题 7：初始 loss 基准值错误 {#P7}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟢 概念纠正 |
| **问题类型** | 概念性 |
| **发现阶段** | 辩论工作流批判审查 |

**症状**：初始 loss 标准被设为 `ln(8192) ≈ 9.01`，Critic 指出这是均匀分布的熵，不适用于自然语言。

**正确理解**：
- 自然语言服从 Zipf 分布，熵远低于均匀分布
- 实际初始 loss 应在 **6.0-7.5**
- < 5.0 → 数据过于重复或有信息泄露
- > 9.0 → tokenizer 接近字符级分割

**合成数据验证**：初始 loss=8.21（偏高），因为模板重复导致 token 分布更均匀。

**正式数据验证**：初始 loss=9.07（也在偏高区），因为纯中文数据 + 8192 词表覆盖良好但 token 分布仍然相对均匀。

**经验教训**：理论公式（ln(vocab)）不替代实验基准。每次换数据/tokenizer 后，直接看第一批 batch 的 loss 作为该数据集的真实基线。

---

### 问题 8：`min_text_len=100` 默认值不适合短文本 {#P8}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 配置问题 |
| **问题类型** | 工程性 |
| **发现阶段** | 合成数据预处理 |

**症状**：合成文本平均 38 字符，默认 `min_text_len=100` 过滤掉 100% 的数据。

**解决方案**：合成数据场景显式传 `--min_text_len 20`。正式数据（百科文章几百到几千字）不受影响。

**经验教训**：默认值应该匹配数据特征。把关键过滤阈值做成**命令行参数**而非硬编码。

**耗时**：2 分钟

---

### 问题 9：确定性验证中 DataLoader 没 shuffle {#P9}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟢 主动发现 |
| **问题类型** | 概念性 |
| **发现阶段** | 确定性验证脚本 |

**说明**：两次固定 seed=42 的 10-step 训练，loss 差精确为 0。这是因为：

1. `DataLoader(shuffle=False)` —— 没有随机化
2. `torch.backends.cudnn.deterministic = True` —— 禁用了 CUDA 非确定性优化
3. 没有 Dropout（`dropout=0.0`）

这是在可控条件下验证了管线确定性的「理想情况」。未来引入 shuffle + DataLoader workers 后，确定性需要额外的 `torch.Generator` 参数。

**经验教训**：确定性是所有 A/B 实验的前提。先证明 seed 一致 → loss 一致，再引入变量。

---

## 三、数据获取阶段

### 问题 10：Docker Hub 和 pip 源被墙 {#P10}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性（暂时） |
| **问题类型** | 工程性（网络） |
| **发现阶段** | Docker 镜像构建 |

**症状**：
- `docker pull pytorch/pytorch:2.6.0-cuda12.4` → timeout
- `docker pull nvidia/cuda:12.0-base` → timeout  
- Tsinghua pip mirror 没有 `tokenizers`/`datasets` 包

**诊断过程**：
1. 检查 Docker daemon 配置：`default-runtime: nvidia` ✅，GPU passthrough 正常
2. 搜索已有镜像：发现本地使用了 `docker.1ms.run`、`docker.m.daocloud.io`、`jx-gpu.gcu.edu.cn:8001` 等镜像代理
3. 测试各镜像：`docker.1ms.run` 可拉取 PyTorch 官方镜像
4. pip 默认源在容器内可用（只是 Tsinghua 源缺包）

**解决方案**：用 `docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` 成功构建。

**耗时**：20 分钟（搜索 + 测试多路镜像）

---

### 问题 11：CUDA 12.4 镜像与驱动 525 不兼容 {#P11}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性（硬件兼容性） |
| **发现阶段** | Docker 容器 GPU 验证 |

**症状**：`Error 804: forward compatibility was attempted on non supported HW`

**原因**：服务器 NVIDIA 驱动 525.60.11 最高支持 CUDA 12.0，PyTorch 2.6.0+cu124 镜像需要更新的驱动。

**解决方案**：
1. 尝试拉 CUDA 12.1 镜像 → 不存在于镜像市场
2. 换 CUDA 12.0 基础镜像 + pip 装 PyTorch cu118 → 需要额外安装 Python
3. **最终方案**：放弃 Docker，直接用宿主 conda 环境

**为什么 conda 能跑但 Docker 不行？** 宿主机 conda 通过 `libcuda.so` 直接调用驱动，PyTorch cu124 的 forward compatibility 由驱动层解析。Docker 通过 `nvidia-container-runtime` 注入驱动，版本感知更严格——driver 525 的 CUDA API 级别 = 12.0，cu124 镜像要求 ≥12.2。

**经验教训**：
- `nvidia-smi` 显示的 CUDA Version 是驱动支持的最高 CUDA API 版本，**不是**已安装的 CUDA 版本
- 不要为环境一致性强行上 Docker——宿主 conda 如果已经验证可用，就是最简单的方案
- 镜像选择原则：CUDA version ≤ driver `nvidia-smi` 显示的版本

**耗时**：45 分钟（拉镜像 + 多次构建 + 最终放弃）

---

### 问题 12：pyarrow 编译失败（系统太老） {#P12}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性（编译环境） |
| **发现阶段** | 安装项目依赖 |

**症状**：`pip install pyarrow` → `Failed to build 'pyarrow' when installing build dependencies for pyarrow` → 需要编译 C++，服务器 GCC/libstdc++ 版本太老。

**解决方案**：`conda install -c conda-forge pyarrow datasets` —— conda-forge 提供预编译二进制，完全绕过编译。

**经验教训**：
- **pip 的 wheel 不是万能的**：某些包（尤其是依赖 C/C++ 扩展的）pip 可能没有对应你平台的 wheel，会 fallback 到源码编译
- **分工原则**：纯 Python 包用 pip（快），需要 C 扩展的用 conda-forge（预编译，零依赖地狱）
- `conda list | grep <package>` 比 pip 更快地确认某个包是否已安装

**耗时**：15 分钟（两次尝试 pip + 切到 conda）

---

### 问题 13：HuggingFace 被墙，部分数据集 DNS 不通 {#P13}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 数据缺口（不阻塞总流程） |
| **问题类型** | 工程性（网络） |
| **发现阶段** | 数据下载 |

**症状**：
- `huggingface.co` 直连 → DNS 失败
- `hf-mirror.com` → 中文维基百科 ✅（146K 文档, 436MB）
- `hf-mirror.com` → FineWeb-Edu ❌（仍然 DNS 失败）
- `hf-mirror.com` → mc4-zh ❌（仍然 DNS 失败）

**当前状态**：只有中文维基百科数据，缺少英文和多样化中文网络文本。

**影响**：训练数据只有纯中文百科 → 模型没见过英文、代码、对话等多样化文本 → 纯中文 PPL 高且泛化差。

**可能的后续方案**：
1. 用服务器已有的 HTTP 代理配置 `HF_ENDPOINT` 环境变量
2. 从本地 PC 下载数据再 scp 上传
3. 换用其他国内可访问的中文语料源（如 CLUECorpus2020、WuDaoCorpora 的采样）

**耗时**：10 分钟（尝试 + 接受结果）

---

## 四、训练阶段

### 问题 14：SSH 超时导致长任务被杀 {#P14}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性（运维） |
| **发现阶段** | Tokenizer 训练中途断开 |

**症状**：长时间运行的命令被 `Connection reset by peer` 中断，SSH 客户端超时后进程也被 SIGHUP。

**解决方案**：用 `nohup` + 后台 + 输出重定向：
```bash
nohup python scripts/train.py > /tmp/training.log 2>&1 &
```

**配套技巧**：
- 检查进度：`tail -20 /tmp/training.log`
- 检查进程是否存活：`ps aux | grep <script_name>`
- 查看 GPU 使用：`nvidia-smi`

**经验教训**：所有预计运行超过 1 分钟的任务都应该用 nohup 或 tmux/screen。

**耗时**：5 分钟

---

### 问题 15：GPU 被其他用户占用 {#P15}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 非阻塞（换卡即可） |
| **问题类型** | 工程性（资源共享） |
| **发现阶段** | 训练启动时 OOM |

**症状**：`CUDA out of memory`，但自己还没开始训——GPU 0 有 20.76 GiB 被占用。

**诊断**：`nvidia-smi` 显示 GPU 0 和 GPU 7 有 `[Not Found]` 进程（已退出的僵尸进程残留显存），GPU 0 还有 `liumi` 用户的 CSCD 训练。

**解决方案**：
1. 杀掉自己的僵尸进程：`kill -9 <pid>`
2. 别人的进程不碰，换到空闲 GPU（1/2/3/4/5/6/8 都空闲）

**经验教训**：
- 共享服务器上，**训练开始前第一件事就是 `nvidia-smi`**
- 程序退出后 GPU 显存可能不会立即释放（CUDA context 残留），需要 `kill -9` 或等几秒
- 多人环境要约定 GPU 使用规范（用 `CUDA_VISIBLE_DEVICES` 限定）

**耗时**：5 分钟

---

### 问题 16：DataLoader 多进程死锁 {#P16}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性（PyTorch multiprocessing） |
| **发现阶段** | 单卡训练启动 |

**症状**：`Train: 64,243 samples` 之后没有任何输出，Python 进程在运行但 GPU 利用率为 0。

**原因**：`DataLoader(num_workers=4)` 在同一个 Python 进程 fork 出 4 个子进程。如果父进程和子进程之间存在共享内存竞争、文件描述符冲突、或者 CUDA 上下文复制问题，子进程会永久阻塞在 `__iter__` 上。

**解决方案**：`num_workers=0` —— 数据加载在主进程同步完成。对 14M 模型的小 batch (32×2048=64KB)，I/O 时间远小于 GPU 计算时间，`num_workers=0` 不会成为瓶颈。

**权衡**：
- `num_workers=0`：简单可靠，适合小模型/小数据
- `num_workers=4`：需要配合 `persistent_workers=True` + GPU 计算时间 >> I/O 时才值得

**经验教训**：多进程 DataLoader 是 PyTorch 最常见的隐蔽死锁源。排查标准流程：
1. 先换 `num_workers=0`——如果好了，就是多进程问题
2. 加 `torch.multiprocessing.set_start_method('spawn')`（对 CUDA 更友好）
3. 确认数据类不包含不可 pickle 的对象

**耗时**：10 分钟

---

### 问题 17：DDP 训练 rank 1 在完成时崩 {#P17}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 训练已完成，不影响结果 |
| **问题类型** | 工程性（分布式同步） |
| **发现阶段** | DDP 训练结束 |

**症状**：500/500 步完成，eval 和 `Done!` 都正常打印，但 torchrun 报 `ChildFailedError: rank 1 exitcode 1`。

**推测原因**：`save_checkpoint` 只在 rank 0 执行 `ckpt_dir.mkdir` + 写文件。rank 1 在 500 步循环结束后执行到 `train_loader.sampler.set_epoch(epoch)` 时（循环条件判断和实际执行之间），rank 0 已完成并退出 DDP group，rank 1 的 NCCL 通信超时。

**影响**：checkpoint 和 run_summary.json 已正确保存，训练结果未丢失。

**修复方向**：在 `dist.destroy_process_group()` 前加 `dist.barrier()` 确保所有 rank 同步退出。

**经验教训**：DDP 程序中，所有需要同步的操作（save、eval、exit）前后都应加 barrier。任何 rank 提前退出都会导致其他 rank NCCL 超时。

**耗时**：5 分钟（事后诊断，不需要重新训练）

---

### 问题 18：训练不收敛 — Loss 在 step 100+ 后几乎持平 {#P18}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 核心问题（待解决） |
| **问题类型** | 概念性（训练策略） |
| **发现阶段** | DDP 训练 eval |

**症状**：

| 指标 | 值 | 评价 |
|------|-----|------|
| 初始 loss | 9.07 | 合理（纯中文+8192 词表） |
| Step 100 loss | 7.70 | 降了 1.37 ✅ |
| Step 500 loss | 7.70 | 和 step 100 几乎一样 ❌ |
| Eval loss 曲线 | 7.704 → 7.702 → 7.701 → 7.700 → 7.699 | 400 步降 0.005 ❌ |
| 最终 PPL | 2206 | 预期 30-60 ❌ |

**根因分析**（三级原因）：

| 层级 | 原因 | 证据 |
|------|------|------|
| **直接原因** | Cosine decay 的「有效窗口」太短，LR 在 step 300 后接近 0 | 日志 `lr` 从 1.20e-03 指数衰减到 1.01e-06 |
| **中层原因** | 500 步太少 × cosine decay 的数学特性：LR 在 warmup 后立刻进入快速衰减区，损失函数还没到平原就被迫停下了 | step 100-200 loss 降幅显著大于 step 200-500 |
| **深层原因** | 数据量 133M tokens 只有 Chinchilla 最优 (280M) 的 47%，模型没吃够 | 数据只来自中文维基百科 |

**这不是代码 bug**——代码正确完成了训练。这是训练配置（LR schedule、总步数、数据量）没有对齐的问题。

**待讨论的修复方案（见下一节）**

---

## 已解决问题速查表

| # | 问题 | 类别 | 解决方案 | 耗时 |
|---|------|------|---------|------|
| P1 | RoPE 维度广播失败 | 🔴 工程 | repeat_interleave 到 d_head | 15min |
| P2 | 循环导入 | 🔴 工程 | 类定义移到子模块 | 2min |
| P3 | Phase 2 参数超预算 | 🟡 概念 | 重新计算规格 | 5min |
| P4 | GitHub 幽灵贡献者 | 🟡 工程 | 修 git config + 删库重建 | 30min |
| P5 | PYTHONPATH 缺失 | 🔴 工程 | scripts 加 sys.path | 2min |
| P6 | 合成数据中文 UNK | 🟢 预期 | 等待正式数据 | 5min |
| P7 | 初始 loss 基准错误 | 🟢 概念 | ln(vocab) → 实测 | — |
| P8 | min_text_len 过严 | 🟡 工程 | 做成 CLI 参数 | 2min |
| P10 | Docker Hub 被墙 | 🔴 工程 | docker.1ms.run 镜像 | 20min |
| P11 | CUDA 12.4 vs 驱动 525 | 🔴 工程 | 放弃 Docker，用 conda | 45min |
| P12 | pyarrow 编译失败 | 🔴 工程 | conda-forge 预编译 | 15min |
| P13 | HF 部分数据集 DNS 不通 | 🟡 工程 | 接受中文维基 | 10min |
| P14 | SSH 超时断连 | 🔴 工程 | nohup | 5min |
| P15 | GPU 被占用 | 🟡 工程 | 换空闲 GPU | 5min |
| P16 | DataLoader 多进程死锁 | 🔴 工程 | num_workers=0 | 10min |
| P17 | DDP rank 1 退出崩 | 🟡 工程 | 加 barrier | 5min |
| — **总计** | **17 个问题** | | | **~3 小时** |

---

## 五、已解决：根因定位（P18 → 诊断链条）

### P18 诊断过程：loss 锁死在 7.70 的根因

**最初假设**：Cosine decay LR schedule 过早衰减导致停止学习。

**推翻过程**：

| 实验 | 模型 | LR schedule | 最终 loss |
|------|------|-------------|-----------|
| v1 DDP | 14M | Cosine decay | 7.70 |
| v2 DDP | 14M | WSD (constant 8e-4) | 7.70 |
| Phase 0 1M | 1M | Constant 3e-4~5e-3 | 全部 7.70 |

**关键发现**：无论模型大小、LR调度、LR值，所有配置都在 loss 7.70 处同时锁死。1M 和 14M 行为完全相同——说明瓶颈不在模型架构或优化器。

**五项诊断全部正常**：
- ✅ 梯度非零（fp16/bf16/fp32 均正常）
- ✅ 数据编码无乱码
- ✅ Loss 计算正确（ignore_index、labels 对齐）
- ✅ Causal mask 方向正确
- ✅ 金丝雀测试：5 条重复句子 → PPL 1.0（Pipeline 健康）

**数据缩放测试**：

| 文档数 | 最终 loss | PPL |
|--------|----------|-----|
| 10 | 6.34 | 569 |
| 100 | 6.92 | 1009 |
| 500 | 7.30 | 1481 |
| 1000 | 7.37 | 1595 |
| 10000 | 7.36 | 1567 |
| 50000 | 7.39 | 1622 |

即使只有 10 篇文档，PPL 也高达 569——说明问题不在 token 总量，而在 token 分布。

### ✅ 根因确认：Token 分布的罕见性瓶颈

**机制**：百科文档每篇包含独有的地名、人名、年代、专业术语。146K 篇文档贡献了 8192 个全部词表 token，但 80%+ 的 token 类型在全集中只出现 1-2 次。模型面对无法从统计上学习的罕见 token，退而求其次到高频 token 的最简预测，loss 锁死在 7.70。

**三层验证证据**：

1. **金丝雀测试**：5 条句子×200 次重复 → 每个 token 出现数百次 → PPL 1.0
2. **数据缩放测试**：10 篇文档已经足以让 loss 撞墙（PPL 569）
3. **方案 E/F/G 对比**（见下文）

### 方案 E/F/G：三种数据策略对比

| 方案 | 策略 | 数据规模 | Token 类型 | 重复度 | 最终 PPL |
|------|------|---------|-----------|--------|---------|
| **E** | 精选百科 1000篇×5epoch | 15K tokens | ~2000 unique | 中 | **84** |
| **F** | 模板合成 5000条 | 40K tokens | ~100 unique | 极高 | **1** ⭐ |
| **G** | 短文本摘要 5000条 | 95K tokens | ~1500 unique | 中 | **78** |
| 原始 | 全量百科 146K篇 | 133M tokens | 8192 unique | 极低 | **2200** |

**结论**：Token 重复度是 PPL 的第一驱动力。方案 F 用 50 种实体 + 60 种模板，token 类型控制在 ~100，每个 token 出现数百次——PPL 收敛到 1.0。方案 E/G 通过限制文档范围提升了重复度，PPL 从 2200 降到 78-84，验证了方向正确但需要更多 epoch。

**最终方案**：基于 F 的模板合成思路，扩展到真实训练规模——50 种模板覆盖 6 个领域，实体池 500+，每模板变异 500 次，生成约 300K-500K tokens 高质量中英混合数据。配合 5-10 epoch 训练和 WSD LR schedule。

---

## 六、经验总结

### 最有价值的教训

1. **合成数据冒烟测试是 ROI 最高的步骤**。30 分钟暴露 5 个集成问题，省了至少 3 天。

2. **先验证环境，再写代码**。Docker 用了 45 分钟才放弃，如果一开始就检查 `nvidia-smi` 驱动版本 + conda 已有包，5 分钟就能确认环境 OK。

3. **Chinchilla 对小模型不适用，但 47% 最优数据量仍然太少了**。14M 参数应该训练 280M+ tokens。

4. **Cosine decay 在短训练中会提前「罢工」**。对 < 1000 步的训练，constant LR 或 linear decay 更合适。

5. **DDP 4 卡的加速比接近线性（3.8x）**。唯一代价是多了一个 barrier 同步的坑。

### 下次起一个新项目时的检查清单

```
□ git config user.name / user.email 确认
□ nvidia-smi 确认 GPU 可用性
□ 确认 Python 环境 + 关键包版本
□ 10MB 合成数据跑通全链路
□ 固定 seed 确定性验证
□ 第一个真实数据 batch → 记录 baseline loss（不是 ln(vocab)）
□ nohup 或 tmux 启动长任务
□ 训练前确认 LR schedule 的「有效窗口」占总步数 > 50%
```


---

## 七、第二阶段：100M 模型预训练（2026-06-26 晚 ~ 06-27）

### 问题 19：OpenSSH 10.0 无法连接老服务器 {#P19}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性（SSH 协议兼容） |
| **发现阶段** | 训练中断后重连 |

**症状**：`Connection reset by 10.1.36.65 port 22`，所有 SSH 命令全部失败。

**原因**：本地 Windows 的 OpenSSH 10.0 默认启用了 post-quantum KEX 算法（`sntrup761x25519-sha512`），服务器 CentOS 7 的 sshd 版本不支持。

**解决方案**：在 `~/.ssh/config` 添加旧算法兼容：
```
Host school
  HostName 10.1.36.65
  User b23113_
  KexAlgorithms diffie-hellman-group14-sha256,diffie-hellman-group-exchange-sha256,ecdh-sha2-nistp256
  HostKeyAlgorithms ssh-ed25519,rsa-sha2-512,rsa-sha2-256
  PubkeyAcceptedKeyTypes ssh-ed25519,rsa-sha2-512
```

**经验教训**：OpenSSH 9.0+ 的默认 KEX 变化是已知的兼容性问题。在老旧服务器环境中，提前配好 `~/.ssh/config` 可以避免误判为"服务器宕机"。

**耗时**：40 分钟（多次误判为宕机）

---

### 问题 20：100M 模型冒烟测试——深窄架构验证 {#P20}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟢 主动验证 |
| **问题类型** | 架构验证 |

**配置**：d=512, L=24, GQA 2:1, QK-Norm, ~99M params

**五项检查全部通过**：
- Forward pass: 3.2GB VRAM, 无 NaN/Inf ✅
- Backward pass: 总 6.2GB, 24 层全部有梯度 ✅
- 残差流分析: 深层 L18-23 方差 0.77-0.92，有效秩 188-191，无瓶颈 ✅
- 梯度健康: Late/early ratio=0.214，梯度平衡 ✅
- 200 步稳定性: loss 9.16→0.11, grad norm 0.23 ✅

**结论**：深窄架构（depth/width=0.047）在 100M 规模下稳定可用。

---

### 问题 21：P1 100M 预训练——低 PPL 但生成全是垃圾 {#P21}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 核心问题 |
| **问题类型** | 概念性（数据分布 × CE loss 机制） |
| **发现阶段** | 0.5B tokens 训练完成后生成测试 |

**训练配置**：
- 50K 模板 + 10K wiki，0.5B tokens
- 4-GPU DDP，loss 9.14→0.008，VAL PPL=2
- 训练时间：~2 小时

**症状**：

```
Prompt: 人工智能是
Output: 人工智能是保加利亞的城鎮,位於該國東南部,距離首府斯維爾約公里...  ← 垃圾

Prompt: 北京是中国的
Output: 北京是中国的国际的三个车团体,它采用巴黎、俊、城、驢...  ← 垃圾
```

**PPL=2 但所有输出都是胡言乱语。** 模型只是背下了 wiki 中随机的地名/人名片段。

**根因分析**（学术文献验证）：

学术上称为 **"Marginalization by Cross-Entropy Loss"**（Turumtaev 2025）或 **"Hyperfitting"**（Carlsson et al. 2024）：

1. CE Loss 对罕见 token 产生不对称的梯度信号：
   - **正向信号（稀有）**：当 token "保加利亞"出现时，拉近 embedding 和上下文。出现 2 次。
   - **负向信号（每次都有）**：每次预测其他 token 时，推开 "保加利亞" embedding。出现 500M 次。
   - **结果**：罕见 token 的 embedding 被负向梯度压制成纯噪声。

2. Wiki 的 10K 条目引入了大量「出现 1-2 次」的罕见地名/人名/术语。

3. 模板数据（50K 条，高频 token）PPL 学得很好（loss=0.008），但一旦采样到 wiki 引入的罕见 token，embedding 是噪声 → 输出垃圾。

**教训**：PPL 在小模型 + 稀疏数据场景下是完全欺骗性的指标。模板数据的 train-test 泄漏让 PPL 虚低。

---

### 问题 22：方案 1 模板引擎——4 次崩溃修复 {#P22}

| 属性 | 内容 |
|------|------|
| **严重度** | 🔴 阻塞性 |
| **问题类型** | 工程性（代码 bug） |

**四次崩溃记录**：

| 尝试 | 错误 | 根因 | 修复 |
|------|------|------|------|
| 1 | `NameError: name 'a' is not defined` in `description()` | `{a}` 用了未绑定的局部变量 | 加 `an = self.pick(WORD["animal"])` |
| 2 | `{s}` 未定义 in `dialogue()` | 残留在模板字符串中的旧变量 | 删除 `{s}` |
| 3 | `NameError: name 'pick' is not defined` in `procedure()` | `pick(WORD...)` 写了裸调用 | `self.pick(WORD...)` |
| 4 | `NameError: name 'pick' is not defined` in `definition()` | 同样问题，多处函数 | **批量正则替换**：`pick(WORD` → `self.pick(WORD` |

**教训**：模板引擎的 `{var}` 在 f-string 中如果变量未绑定会直接 NameError。最佳实践是写一个 `_v(rng)` 统一提取所有变量，避免逐函数手动声明。

---

### 问题 23：方案 2 MiniMind 真实数据——突破性成功 {#P23}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟢 实验成功 |

**数据来源**：服务器已有的 `~/minimind-master/dataset/pretrain_t2t_mini.jsonl`
- 1.2GB，127 万行真实中文 Web 文本
- 格式：`{"text": "..."}` JSONL
- 混合了指令、对话、描述、Q&A、闲聊等多种类型

**训练配置**：同样 100M 模型（d=512 L=24），500K 文档采样，82M tokens，3 epoch

**结果**：

| 指标 | P1 (模板+wiki) | P2 Plan B (真数据) |
|------|---------------|-------------------|
| VAL PPL | 2 | **7** |
| "人工智能是" | "保加利亞的城鎮…" ❌ | "当今科技领域的热点之一" ✅ |
| "北京是中国的" | "国际的三个车团体" ❌ | "首都，位于中国的北部" ✅ |
| "什么是机器学习" | garbage ❌ | "人工智能的分支，从数据中学习" ✅ |
| "春天来了" | random wiki fragments ❌ | "大地充满了生机" ✅ |

**全部 6 个测试 prompt 正确、连贯。**

**核心结论**：PPL=7 > PPL=2 但生成质量差 100 倍。证明 PPL 不是好坏的指标——**数据分布的自然性才是**。同样 100M 模型，只换数据，效果翻天覆地。

---

### 问题 24：方案对比——Plan A vs Plan B 对决 {#P24}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟢 实验结论 |

| 维度 | Plan A (模板引擎) | Plan B (真数据) |
|------|------------------|----------------|
| 数据源 | 500+ 句式模板, 1355 unique tokens | MiniMind pretrain_t2t_mini, 1.2GB |
| VAL PPL | 2 | 7 |
| 生成 "人工智能是" | "理论的道德。亲情更关注习俗…" ❌ | "当今科技领域的热点之一…" ✅ |
| 生成 "北京是中国的" | "正确步骤：第一步，阅读说明…" ❌ | "首都，位于中国的北部…" ✅ |
| 生成 "机器学习" | 幻觉成店员对话 ❌ | "AI的分支，从数据学习…" ✅ |
| 语义连贯性 | 无 | 有 |
| 话题逻辑 | 无 | 有 |

**结论**：真实数据的语义连贯性和话题逻辑是模板无法模拟的。即使 500 个句式也不够。Plan B 完胜。

---

### 问题 25：DDP 退出时 SIGABRT {#P25}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 非阻塞（训练已完成，checkpoint 已保存） |
| **问题类型** | 工程性（分布式同步） |

**症状**：训练完成最后几十步，rank 0 收到 `Signal 6 (SIGABRT)`。

**根因**：`dist.destroy_process_group()` 之前没有 `dist.barrier()`。当 rank 0 完成训练循环后立即执行到 save/exit，其他 rank 还在循环中 → NCCL 超时 → SIGABRT。

**影响**：checkpoint 在训练中每 N 步保存一次，中间 checkpoint 完好。final checkpoint 未生成。

**修复**：在保存和退出前加 `dist.barrier()`。或者改用弹性训练框架的优雅退出机制。

**经验教训**：这是 P1/P2 阶段发生 3 次的重复问题（P17, Plan A, Plan B）。DDP 程序的退出同步是最容易忽视的工程细节。

**耗时**：0（checkpoint 已保存，不需要重新训练）

---

### 问题 26：ModelScope 数据集全部 404 {#P26}

| 属性 | 内容 |
|------|------|
| **严重度** | 🟡 非阻塞（有替代方案） |
| **问题类型** | 工程性（网络/数据源） |

**尝试的数据集**：
- `opencsg/Fineweb-Edu-Chinese-V2.2` → 404
- `BAAI/CCI3-HQ` → 404
- `Tele-AI/TeleChat-PTD` → 404
- `damo/nlp_wikipedia-zh_pretrain` → 404

**解决方案**：直接使用服务器已有的 MiniMind 数据集（`~/minimind-master/dataset/`），无需下载。

**经验教训**：在共享服务器上，先搜索已有数据资源（`find`）比盲目下载快得多。

---

## 八、经验总结（第二阶段新增）

### 最有价值的教训

6. **PPL 是欺骗性指标——尤其是在小模型 + 稀疏数据场景。** P1 的 PPL=2 理论上"完美"，实际生成全是垃圾。P2 的 PPL=7 生成质量好 100 倍。**永远用生成测试验证模型，不要只看 PPL。**

7. **罕见 token marginalization 是小模型的隐形杀手。** CE loss 的负向梯度在 500M 步中累计，把只出现 1-2 次的 token embedding 压成噪声。这不是参数量问题——1B 模型面对同样数据也会有问题。MiniMind 用 10B tokens 才让模型学到有意义的表示。

8. **真实数据的"低 PPL"比模板数据的"高 PPL"有价值得多。** 模板 PPL=2 但生成垃圾，真实数据 PPL=7 但生成正确。因为真实数据的 token 分布和下游任务一致。

9. **模板引擎最多能教句法，教不了语义。** 500+ 句式、5000+ 词汇的模板引擎能让模型学会句法填空，但无法产生有逻辑的跨句语义连贯。这个 gap 是模板方法的天花板。

10. **共享服务器上的已有资源是金矿。** MiniMind 的 1.2GB 预训练数据就在服务器上，省去了下载、清洗、格式化的全部工作。先 `find` 再 `download`。

### 下次启动 100M 训练的检查清单补充

```
□ SSH config 配好 KexAlgorithms 兼容（老服务器）
□ 先 find 服务器上已有的数据集
□ 预训练数据必须包含足够比例的"真实自然文本"
□ 不要只用 PPL 做评估——必须跑生成测试
□ 模板数据可以辅助，但不能超过 30%
□ DDP 脚本加 dist.barrier() 在退出前
```
