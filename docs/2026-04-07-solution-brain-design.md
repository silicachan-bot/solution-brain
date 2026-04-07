# solution-brain 设计文档

> 语言模式提取、存储与注入系统——为聊天模型构建独立人格

## 1. 目标

从 B 站群体评论中提取通用的语言模式（句式、语气、用词习惯等），结构化存储后通过提示词注入引导强基础模型（API 调用）的聊天风格，使其具备独立人格。

**核心原则：** 人格 ≈ 一组可枚举、可检索的语言行为模式，而不是一段固定的 system prompt。

**MVP 目标：** 全流程跑通 + 架构稳固。后续加功能在模块内部改，不需要重构框架。

## 2. 架构概览

Pipeline 架构，模块间通过明确的数据契约衔接：

```
[数据源] → [摄入] → [提取] → [存储] → [检索/组装] → [API调用]
  ingest     extract    store     compose       api
```

不做微调，纯提示词注入路线。调用强 API 模型（如 Claude/GPT）。

## 3. 项目结构

```
solution-brain/
├── pyproject.toml                # uv 项目，Python 3.11
├── src/brain/
│   ├── __init__.py
│   ├── ingest/                   # 数据摄入模块
│   │   ├── __init__.py
│   │   ├── reader.py             # 读取外部数据源（SQLite 等）
│   │   ├── cleaner.py            # 文本清洗、去噪、分批
│   │   └── state.py              # 水位线管理（记录处理进度）
│   ├── extract/                  # 模式提取模块
│   │   ├── __init__.py
│   │   ├── chunker.py            # 评论分块 + 基础清洗去重
│   │   └── refiner.py            # LLM 提取 + 跨块/跨视频去重合并
│   ├── store/                    # 存储与检索模块
│   │   ├── __init__.py
│   │   ├── pattern_db.py         # PatternCard 持久化（向量 DB + 元数据）
│   │   └── retriever.py          # 按语义相似度检索相关模式
│   ├── compose/                  # 上下文组装模块
│   │   ├── __init__.py
│   │   ├── menu.py               # 检索相关模式 → 生成"菜单"列表
│   │   ├── tools.py              # inspect_pattern 工具定义 + 调用处理
│   │   └── templates/
│   │       └── system.txt        # system prompt 模板
│   └── api/                      # 对外接口（后期实现）
│       ├── __init__.py
│       └── service.py            # FastAPI 应用
├── scripts/
│   ├── run_pipeline.py           # 全流程编排入口（支持增量/全量）
│   └── run_extract.py            # 单独跑提取（调试用）
├── tests/
└── data/                         # 本地数据（gitignored）
    └── state.json                # 处理进度状态
```

### 模块间数据契约

| 模块 | 输入 | 输出 |
|------|------|------|
| `ingest` | 外部数据源（SQLite） | `list[CleanedComment]` |
| `extract.chunker` | `list[CleanedComment]`（单视频） | `list[list[str]]`（评论块） |
| `extract.refiner` | 多个评论块 + 已有模式摘要 | `list[PatternCard]`（新增/更新） |
| `store` | `PatternCard` | 持久化 + 检索接口 |
| `compose` | 对话上下文 | system prompt + tools 定义 |

## 4. 核心数据模型

### PatternCard

```python
@dataclass
class PatternCard:
    id: str                       # 唯一标识（UUID）
    title: str                    # 简短标题，如 "好家伙式吐槽"
    description: str              # 模式描述：是什么、什么时候用、传达什么感觉
    template: str                 # 模板句，含 [A] [B] 占位符
    examples: list[str]           # 2-5 个填充后的真实例句
    frequency: FrequencyProfile   # 时间窗口频率
    source: str                   # 数据来源标识
    created_at: datetime
    updated_at: datetime
```

不设分类字段。检索完全依赖向量相似度，人工浏览靠 title + description 区分。如果后续发现需要分组，可以随时加。

### FrequencyProfile

```python
@dataclass
class FrequencyProfile:
    recent: int       # 近 3 个月出现次数
    medium: int       # 近 6 个月
    long_term: int    # 近 2 年
    total: int        # 全部

    @property
    def freshness(self) -> float:
        """趋势越上升 → freshness 越高"""
        if self.total == 0:
            return 0.0
        recent_ratio = self.recent / max(self.total, 1)
        medium_ratio = self.medium / max(self.total, 1)
        return recent_ratio * 0.5 + medium_ratio * 0.3 + min(self.total / 500, 1.0) * 0.2
```

新提取到已有模式时，对应窗口的频率递增，freshness 自动重算。过时的模式自然衰减。

### PatternCard 示例

```yaml
id: "pat-a3f2c1"
title: "好家伙式吐槽"
description: "对出乎意料的事情表达又无奈又好笑的吐槽。省略号制造停顿，'好家伙'点睛。常见于看到离谱操作时的反应。"
template: "[A]...好家伙..."
examples:
  - "这也行...好家伙..."
  - "又双叒叕来了...好家伙..."
  - "直接满分...好家伙..."
frequency:
  recent: 45      # 近 3 个月
  medium: 89      # 近 6 个月
  long_term: 200  # 近 2 年
  total: 350
source: "bilibili_gaming_zone"
```

## 5. 提取流程

```
评论分块 → 每块交给 LLM 找模式 → 跨块去重合并 → 跨视频去重合并 → 入库
```

### 阶段一：分块（`extract.chunker`）

逐视频处理，将评论切成适合 LLM 处理的小块：

```python
def chunk_comments(comments: list[CleanedComment], chunk_size=50) -> list[list[str]]:
    """把一个视频的评论分成若干块，每块约 50 条"""
```

分块前做极简预处理：
- 去掉太短（<5 字）或纯表情的评论
- 去掉完全重复的评论（只保留一条）

不做统计分析、不做骨架提取——把模式发现完全交给 LLM。

### 阶段二：LLM 提取 + 去重合并（`extract.refiner`）

每块评论单独送 LLM，让它直接发现语言模式：

```
以下是 B 站某视频下的 50 条用户评论。
请从中发现值得收录的语言模式——特别是：
- 多人使用的相似句式
- 不像 AI 会自然生成的表达
- 可以替换内容复用的句式模板

对每个发现的模式，输出 JSON：
{ "title": "...", "template": "含[A][B]的模板", "examples": [...], "description": "..." }

如果这批评论中没有值得收录的模式，返回空数组。
```

多个块/多个视频的结果做去重合并：
- 相似模式（embedding 相似度 > 阈值）→ 合并例句、累加频率
- 和已入库模式相似 → 补充已有卡片的例句和频率
- 全新模式 → 创建新 PatternCard 入库

跨视频验证隐式发生——多个视频的 LLM 结果中都出现类似模式，合并后频率自然高，说明是通用的语言模式。

### 成本控制

一个视频约 2000 条有效评论 → 40 块 × 每块约 2000 token ≈ 8 万 input token。使用低成本模型（Claude Haiku / GPT-4o-mini）做提取，成本可控。

### Pipeline 编排

```python
# scripts/run_pipeline.py
def run_pipeline(source, batch_size=10, full=False):
    # 1. 摄入：读取新评论（增量/全量）
    videos = ingest(source, full=full)

    # 2. 逐视频分块 + LLM 提取
    all_patterns = []
    for video in videos:
        chunks = chunker.chunk(video.comments)
        for chunk in chunks:
            patterns = refiner.extract_from_chunk(chunk)
            all_patterns.extend(patterns)

    # 3. 跨块/跨视频去重合并，与已有模式库对比
    existing = store.list_all()
    new_cards, updates = refiner.deduplicate_and_merge(all_patterns, existing)
    store.save(new_cards)
    store.update(updates)

    # 4. 更新水位线
    state.update_watermark(source)
```

支持增量处理：`state.py` 管理水位线，记录每个数据源处理到哪条数据，默认只处理新增部分。

## 6. 存储与检索

### 存储方案

使用 **ChromaDB**（纯 Python，本地文件存储，无需外部服务）：

- PatternCard 的 `description + examples` 文本做 embedding，存入 ChromaDB
- PatternCard 的完整数据以 JSON 形式存储（ChromaDB metadata 或单独的 JSON 文件）
- 向量只是索引，检索到 ID 后取回完整卡片

嵌入模型选项：OpenAI `text-embedding-3-small` 或本地 `sentence-transformers`。

### 检索流程

```python
def get_relevant_patterns(conversation, top_k=8):
    # 1. 取最近几轮对话作为查询文本
    query = format_recent_turns(conversation, max_turns=3)

    # 2. 向量检索，取 top_k * 2 候选
    results = vector_db.query(embed(query), top_k=top_k * 2)

    # 3. 按 freshness 加权重排序
    scored = [
        (card, similarity * 0.6 + card.frequency.freshness * 0.4)
        for card, similarity in results
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [card for card, _ in scored[:top_k]]
```

两个检索因素：语义相关性（向量相似度）+ 时效性（freshness）。

## 7. 上下文注入：渐进式披露

不把所有模式细节塞进 system prompt，而是：

### System prompt 只给"菜单"

```
你有以下语言模式可以使用。回复前你可以查看感兴趣的模式了解详情，
然后决定是否使用。你也完全可以不使用任何模式。

可用模式：
1. [pat-a3f2c1] 好家伙式吐槽 — "[A]...好家伙..."
2. [pat-b7d4e2] 叠字撒娇 — "[A]嘛~[B]啦~"
3. [pat-c9f1a8] 阴阳怪气反问 — "所以[A]是吧，行"
...
```

### 模型通过工具调用查看详情

```python
# 工具定义
{
    "name": "inspect_pattern",
    "description": "查看某个语言模式的详细描述和使用示例。使用任何模式前必须先查看。",
    "parameters": {
        "pattern_id": { "type": "string" }
    }
}

# 返回
{
    "title": "好家伙式吐槽",
    "description": "对出乎意料的事情表达又无奈又好笑的吐槽...",
    "template": "[A]...好家伙...",
    "examples": ["这也行...好家伙...", "又双叒叕来了...好家伙..."]
}
```

### 调用流

```
用户消息 → 模型看到菜单 → 选择感兴趣的模式
→ inspect_pattern(id) → 看到详情 → 决定用/不用 → 生成回复
```

好处：prompt 短、模型主动选择、使用更自然、避免滥用不理解的模式。

## 8. 对外接口（后期实现）

当前 chat-core/bot 使用 OpenAI 兼容的 chat completion API 调用 vLLM，尚未支持 tool use。

对接方案留待模式提取和存储完成后再设计。大方向：
- solution-brain 提供 HTTP 接口供 bot 获取上下文和处理工具调用
- bot 侧需要扩展 llm_client.py 支持 tool use 流程

## 9. 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.11 | 与项目其他子模块一致 |
| 包管理 | uv | 项目统一 |
| 向量数据库 | ChromaDB | 纯 Python，本地存储，零运维 |
| 嵌入模型 | Qwen3-Embedding-4B（API 调用，2048 维） | 质量和成本的折中，API 调用成本低 |
| Web 框架 | FastAPI（后期） | 轻量，异步 |
| 模板引擎 | Jinja2 | prompt 模板渲染 |
| LLM API | OpenAI 兼容 / Claude API | 提取和聊天均通过 API 调用 |
