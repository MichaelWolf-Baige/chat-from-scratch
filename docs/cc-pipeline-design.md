# CommonCrawl 中文数据清洗 Pipeline 方案

> 设计日期: 2026-06-29 | 状态: 方案阶段，待实施

---

## 一、背景与目标

### 项目定位

chat-from-scratch 已经用 Qwen2.5-1.5B 蒸馏了 87K 条数据完成了第一版预训练（PPL=5，生成质量良好）。现在需要走一遍**完整的数据工程流程**——从原始 CommonCrawl 爬取开始，经过清洗、去重、质量过滤，最终产出可用于预训练的高质量中文数据集。

### 核心目的

**积累工程经验**，不是追求最优模型性能。与蒸馏数据的对照实验将直接回答"手工数据 vs 教师模型蒸馏"的效果差异。

### 硬件约束

- 服务器: 9× RTX 3090 (24GB)，数据清洗用 CPU（不占 GPU）
- 本地: Ryzen 7 5700U, 14GB RAM, 72GB 可用磁盘 — 仅用于小样本开发和调试
- 流程: 本地开发 → git push → 服务器 pull → 服务器 CPU 执行

---

## 二、中文适配验证（阶段 0 — 在一切之前）

> ⚠️ **关键决策**: 不直接套用 FineWeb 的英文结论。FineWeb 主要验证英文/欧洲语言，中文部分没有单独评估指标。中文网站 CMS 模板化程度高、大量 `<div>` 嵌套，与英文网站差异大。

### 0.1 正文提取器对比

**候选**: Trafilatura、jusText、readability-lxml

**方法**:
1. 从 3 个不同 snapshot 中随机抽取 200-500 个语言识别为中文的页面 WARC 记录
2. 用三种提取器分别提取正文
3. 人工标注: 完整提取 / 部分丢失 / 严重漏提 / 误保留非正文
4. 统计各提取器的 precision 和 recall
5. **根据实证结果做选择**，不依赖 FineWeb 论文结论

### 0.2 MinHash n-gram 策略对比

**方法**:
1. 手工构造几组已知重复关系的中文文档:
   - 完全相同（复制粘贴）
   - 同一篇文章在不同模板站点渲染
   - 大部分相同但插入了不同推荐/广告
   - 不同文章但共享模板化 footer/header
2. 分别用字符级 5-gram 和 jieba 分词后的 token 3-gram/5-gram 做 MinHash 签名
3. 计算各组已知重复对的 Jaccard 相似度
4. 判断哪种方案能更好地区分真实重复 vs 模板相似

> 中文每个汉字是独立语义单元，字符级 5-gram 仅覆盖 5 个字（约 2-3 个词）。中文模板化内容（如"发布于 2023-12-01 来源：某某新闻"）通常以短语级别重复。

### 0.3 跨 Snapshot 重复率估算

从 2 个不同年份的 snapshot 中各随机抽取 5000 个中文页面，做 MinHash 近似查重:

- 如果重复率 < 5%: 每个 snapshot 独立去重
- 如果重复率 > 5%: snapshot 内先去重，再跨 snapshot 保守去重

---

## 三、总体架构: 5+1 阶段 Pipeline

```
阶段0(中文验证) → 阶段1(数据获取) → 阶段2(提取初筛)
    → 阶段3(质量过滤) → 阶段4(去重) → 阶段5(后处理与导出)
```

**设计原则**:
- 每阶段输出中间文件（JSONL），支持断点续跑
- 每步记录保留率（retention rate），采样检查被过滤的文档
- 所有参数集中管理，Git 版本控制

---

## 四、阶段 1: 数据获取

### 4.1 Snapshot 选择

从 2020-2025 年选 3-5 个 snapshot:

| 优先级 | Snapshot | 年份 | 说明 |
|--------|----------|------|------|
| 1 | CC-MAIN-2023-50 | 2023 | 较新，ChatGPT 后数据 |
| 2 | CC-MAIN-2022-49 | 2022 | ChatGPT 前数据 |
| 3 | CC-MAIN-2021-43 | 2021 | 早期数据 |

### 4.2 下载方式

```python
# datatrove 支持直接从 S3 读取
from datatrove.pipeline.readers import WarcReader
WarcReader("s3://commoncrawl/crawl-data/CC-MAIN-2023-50/")
```

如果国内访问 S3 慢:
- 方案 A: HTTP 镜像 `https://data.commoncrawl.org/`
- 方案 B: 先 `wget` 下载 WARC 到本地，再用 `WarcReader` 读本地文件

### 4.3 WARC 记录级过滤

在正文提取前做元数据过滤，零成本裁剪大量垃圾:

```python
from datatrove.pipeline.filters import BaseFilter

class WarcMetadataFilter(BaseFilter):
    """基于 WARC 记录元数据的预过滤"""
    def filter(self, doc):
        # 1. HTTP 非 200 响应
        if doc.metadata.get("http_status") not in (200, None):
            return False, "non_200_status"

        # 2. 非 HTML 内容（PDF, 图片等）
        content_type = doc.metadata.get("content_type", "")
        if "text/html" not in content_type and content_type:
            return False, "non_html_mime"

        # 3. 响应体过小或过大
        size = len(doc.text) if doc.text else 0
        if size < 1024:              # < 1KB
            return False, "too_small"
        if size > 5 * 1024 * 1024:   # > 5MB
            return False, "too_large"

        return True
```

### 4.4 存储估算

- 一个 snapshot 完整 WARC: ~50-70TB
- 中文占比: ~5-8%
- 全链路清洗后保留率: ~5-15%
- 100M 模型目标: **5-10B token**（约 3-7GB 纯文本）
- 3-5 个 snapshot 足够

---

## 五、阶段 2: 文本提取 + 初筛

### 5.1 URL 过滤

```python
from datatrove.pipeline.filters import URLFilter
URLFilter(block_list=["ut1"])  # UT1 黑名单（成人内容、垃圾域名）
```

### 5.2 文本提取

根据阶段 0.1 实证结果选择最佳提取器:

```python
from datatrove.pipeline.extractors import Trafilatura

# 如果验证中 Trafilatura 表现最好:
Trafilatura(favour_precision=True)

# 如果中文页面 precision 模式漏提严重:
Trafilatura(favour_recall=True)

# 如果 jusText 更好:
from datatrove.pipeline.extractors import JusText
JusText()
```

### 5.3 语言识别

```python
from datatrove.pipeline.filters import LanguageFilter

LanguageFilter(
    languages=["zh"],
    language_threshold=0.65,  # 中文误判率高时可降到 0.5
    backend="ft176",          # FastText lid.176.bin
)
```

---

## 六、阶段 3: 质量过滤

### 6.1 Gopher 重复性过滤

```python
from datatrove.pipeline.filters import GopherRepetitionFilter

GopherRepetitionFilter(
    dup_line_frac=0.3,
    dup_para_frac=0.3,
    dup_line_char_frac=0.2,
    dup_para_char_frac=0.2,
    top_n_grams=((2, 0.2), (3, 0.18), (4, 0.16)),
    dup_n_grams=((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10)),
)
```

### 6.2 Gopher 质量过滤（中文适配版）

```python
from datatrove.pipeline.filters import GopherQualityFilter

CHINESE_STOP_WORDS = [
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如果", "因为", "所以", "但是", "可以", "这个", "那个",
]

GopherQualityFilter(
    min_doc_words=30,
    max_doc_words=100000,
    min_avg_word_length=1,       # 中文放宽
    max_avg_word_length=20,
    max_symbol_word_ratio=0.1,
    max_bullet_lines_ratio=0.9,
    max_ellipsis_lines_ratio=0.3,
    max_non_alpha_words_ratio=0.8,
    min_stop_words=2,
    stop_words=CHINESE_STOP_WORDS,
    language="zh",
)
```

### 6.3 自定义中文质量规则

```python
class ChineseQualityFilter(BaseFilter):
    def filter(self, doc):
        text = doc.text

        # 1. 中文字符数（绝对计数，非比例，避免短文档误判）
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        if chinese_chars < 50:
            return False, "too_few_chinese"

        # 2. 中文占比过低
        if len(text) > 0 and chinese_chars / len(text) < 0.3:
            return False, "low_chinese_ratio"

        # 3. 感叹号密度过高（标题党特征）— 用总文本长度做分母
        exclam_question = (
            text.count('！') + text.count('？') +
            text.count('!') + text.count('?')
        )
        if len(text) > 0 and exclam_question / len(text) > 0.1:
            return False, "too_many_exclam"

        # 4. 行数 + 平均行长检查（防御 SEO 刷词页面）
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 3:
            return False, "too_few_lines"
        if sum(len(l) for l in lines) / len(lines) < 10:
            return False, "seo_short_lines"

        return True
```

### 6.4 中文截断检测

中文网页分页/动态加载导致的截断问题普遍存在:

```python
class ChineseTruncationFilter(BaseFilter):
    """检测正文是否被意外截断"""
    def filter(self, doc):
        text = doc.text.strip()
        if not text:
            return True

        last_50 = text[-50:] if len(text) >= 50 else text
        sentence_ends = {'。', '？', '！', '"', '…', '；', '：', '\n'}
        if not any(c in last_50 for c in sentence_ends):
            return False, "possibly_truncated"
        return True
```

### 6.5 文档质量连续打分

当清洗后数据量超过目标时，需要按质量排序而非随机抽样:

**方案 A（轻量）: FastText 质量分类器**
- 手动标注 500-1000 个文档为 high/medium/low 三档
- 特征: 文档长度分位数、中文占比、重复 n-gram 比例、符号密度、段落数
- 输出置信度作为质量分数

**方案 B（重量）: KenLM 中文困惑度**
- 用 Wikipedia 中文 + 高信噪比 CC 训练 5-gram 语言模型
- 对每文档计算 perplexity
- 低困惑度 = 更符合自然语言分布
- 设上下界: PPL < 10 可能是机器生成模板, > 2000 可能是噪音

**建议**: 先跑通基础 pipeline。数据量 > 目标 3 倍以上再加方案 A。方案 B 实测后判断。

---

## 七、阶段 4: 去重

### 7.1 去重策略

由阶段 0.3 实证结果决定:
- 跨 snapshot 重复率 < 5%: 各自独立去重
- 跨 snapshot 重复率 > 5%: snapshot 内去重 → 跨 snapshot 保守去重

### 7.2 MinHash 参数

根据阶段 0.2 实证选 n-gram 策略:

```python
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupSignature

minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,  # 字符级 5-gram 或分词级 3-gram
)
```

4 阶段流程:
```
Signatures → Buckets → Cluster → Filter
```

### 7.3 PII 过滤（前置到去重之前）

```python
# 必须在 MinHash 签名计算之前执行
# 原因: 同篇文章因不同评论区 PII 会导致 MinHash 签名不匹配
from datatrove.pipeline.formatters import PIIFormatter
PIIFormatter()
```

### 7.4 URL 去重

相同 URL: **保留质量更高的版本**（文本更长、中文占比更高），而非保留最新的（最新版本可能是 404 或改版低质量内容）。

### 7.5 内存尖峰对策

MinHashDedupCluster 在聚类时需加载全量签名:
- N=500万时: ~560MB 签名 + 倒排索引 ≈ 10-20GB
- N > 1000万: 按 URL hash 分 4-8 片，每片内独立 MinHash，边界文档合并去重

### 7.6 不需要的

- ❌ ExactSubstr 后缀数组去重: 内存极大，ROI 极低
- ❌ 句子级去重: 100M 模型数据量不是瓶颈

---

## 八、阶段 5: 后处理与导出

### 8.1 数据分片与格式

```json
{
  "text": "清洗后的文本内容...",
  "metadata": {
    "source": "CC-MAIN-2023-50",
    "url": "http://example.com/article",
    "warc_path": "CC-MAIN-2023-50/segments/.../warc/CC-MAIN-...",
    "warc_offset": 123456,
    "language": "zh",
    "language_score": 0.95
  }
}
```

按 100MB-500MB 分片，每文件一个 JSONL。

### 8.2 Tokenizer 训练

**训练数据构成**: 80% 高质量清洗数据 + 20% 轻度清洗数据

原因: 纯高质量数据会让 tokenizer 缺失罕见字/特殊符号，后续推理时遇到未见字符 fallback 到 UNK。混入部分轻度清洗数据保持覆盖。

**词表大小**: 16K-32K BPE（100M 模型不宜过大，embedding 层参数占比过高）

### 8.3 数据配比

建议最终预训练数据配比:

| 数据源 | 比例 | 说明 |
|--------|------|------|
| 清洗后的 CC 中文 | 70-80% | 主力数据 |
| 中文 Wikipedia | 10-15% | 高质量知识锚点 |
| 中文书籍/对话 | 5-10% | 增加表达多样性 |
| 代码（可选） | 5% | 逻辑推理能力 |

> ⚠️ 此配比为参考值。实际最优配比需消融实验确定。对于 100M 模型，可设计 3-5 组小规模消融（每组 10M 参数，500M token）快速验证。

**总 token 量**: 2-5B token（可扩展到 5-10B，高质量子集可 2-3 次 epoch）

---

## 九、工具链汇总

| 步骤 | 推荐工具 | 替代方案 |
|------|----------|----------|
| Pipeline 编排 | **datatrove** (LocalPipelineExecutor) | 自定义 Shell 脚本 |
| 文本提取 | **阶段 0 实证后决定** | Trafilatura / jusText / readability-lxml |
| 语言识别 | FastText lid.176.bin | GlotLID |
| WARC 元数据过滤 | 自定义 BaseFilter | datatrove GopherURLFilter |
| 去重 | MinHash (datatrove 内置) | datasketch |
| PII 过滤 | PIIFormatter (datatrove) | 正则表达式 |
| 正文分词 | jieba | pkuseg |
| Tokenizer 训练 | HuggingFace tokenizers | sentencepiece |

---

## 十、中文特殊性问题

| 问题 | 解决方案 |
|------|----------|
| 中文无空格分词 | Gopher 过滤器需 jieba 分词或放宽 word 计数参数 |
| 停用词不同 | 用中文高频虚词替换英文停用词 |
| 字符长度计算不同 | 放宽 `avg_word_length` 阈值 |
| C4 终端标点检测不适用 | 关闭 `filter_no_terminal_punct`；新增中文截断检测 |
| MinHash n-gram 选择 | **阶段 0.2 实测后决定**（字符级 vs 分词级） |
| 正文提取器选择 | **阶段 0.1 实测后决定**（不预设 Trafilatura 最优） |

---

## 十一、执行计划

| 阶段 | 时间 | 机器 | 内容 |
|------|------|------|------|
| 0 | 0.5 天 | 本地/服务器 | 三项实证验证 |
| Day 1 | 开发 | 本地 | 搭建 datatrove 环境，下载 1 个 WARC，跑通基础流程 |
| Day 1 末 | — | — | **用 1 个 WARC 的处理时间校正后续日程** |
| Day 2 | 调试 | 本地/服务器 | 中文版 Gopher 过滤器适配、自定义规则调参 |
| Day 3-4 | 运行 | 服务器 CPU | 完整 snapshot 处理（可能 2-4 天/snapshot） |
| Day 5 | 运行 | 服务器 CPU | MinHash 4 阶段去重 + PII + 分片 + Tokenizer |
| Day 6 | 验证 | 服务器 GPU | 数据混合 + 10M 模型小规模训练验证 |
| Day 7+ | 扩展 | 服务器 CPU | 处理更多 snapshot |

> 实际时间取决于 CPU 核数、网络速度、Trafilatura 吞吐量。Day 1 结束后根据实测重新排期。

---

## 十二、不推荐的做法

- ❌ 跳过 Trafilatura 直接用 WET（质量损失已被 FineWeb 证明），**BUT** 如果阶段 0.1 发现 Trafilatura 对中文效果不如预期，可用 WET + 更重的后过滤
- ❌ 盲目做全局去重——先测跨 snapshot 重复率
- ❌ 用 LLM（GPT-4 等）做质量打分——过度工程化
- ❌ 一开始就处理所有 snapshot——先跑通 1 个
- ❌ 跳过阶段 0 的实证验证——英文结论不等于中文实测

---

## 十三、与蒸馏数据的对照实验

完成 CC pipeline 后，设计系统对照:

| 维度 | 蒸馏数据 | CC 手工清洗数据 |
|------|----------|----------------|
| 数据来源 | Qwen2.5-1.5B 生成 | CommonCrawl 公开网页 |
| 处理流程 | Teacher 蒸馏 | 完整 CC pipeline |
| 数据量 | 87K 条，~13M tokens | 目标 2-5B tokens |
| 格式 | 统一对话体 "A:...\n\nB:..." | 自然文本混合 |
| 工程经验 | 蒸馏技术 | 数据工程全栈 |

**对照实验**: 用同一 100M 架构、同一 tokenizer、同一训练配置，比较两套数据的预训练效果。

---

## 十四、参考资料

- **FineWeb** (HuggingFace, 2024): 数据清洗 pipeline 和 datatrove 库
- **DCLM** (DataComp for Language Models, 2024): 数据处理消融实验
- **Dolma** (AI2, 2024): 详细的数据处理文档
- **LLaMA 3 技术报告**: 数据配比和过滤决策
- **Gopher** (DeepMind, 2021): 数据质量过滤方法
- **Chinchilla** (DeepMind, 2022): 计算最优数据量
- **BloombergGPT** (2023): 手动数据工程的价值例证
