# 模块文档总览

本文档目录用于描述 `solution-brain` **当前已经实现的代码结构**，不是目标架构草案。阅读时应以仓库中的实际文件为准。

## 1. 文档范围

当前目录覆盖以下实现模块：

1. `shared`：公共数据模型与配置
2. `ingest`：从 Bilibili SQLite 读取评论、清洗、维护水位线
3. `extract`：评论分块、调用 LLM 提取模式、按标题去重合并
4. `store`：PatternCard 持久化、embedding、检索排序
5. `compose`：菜单构造、system prompt 组装、`inspect_pattern` 工具
6. `pipeline`：把 ingest → extract → store 串起来的命令行入口
7. `viewer`：Streamlit 模式库查看器

不包含尚未落地的独立 API 层；旧设计文档里提到的 `api/` 目录在当前仓库中不存在。

## 2. 代码总入口

当前运行入口主要有两个：

- `scripts/run_pipeline.py`：执行数据提取流水线
- `scripts/streamlit_patterns.py`：启动模式库查看器

核心代码位于：

- `src/brain/`

## 3. 当前模块关系

```text
Bilibili SQLite
    ↓
ingest.reader.list_videos / read_comments
    ↓
ingest.cleaner.clean_comments
    ↓
extract.chunker.chunk_comments
    ↓
extract.refiner.extract_from_chunk
    ↓
extract.refiner.deduplicate_and_merge
    ↓
store.pattern_db.save / update
    ↓
store.retriever.retrieve_patterns
    ↓
compose.menu / compose.assembler / compose.tools
    ↓
viewer 或外部调用方消费
```

## 4. 当前实现中的核心对象

### 4.1 `CleanedComment`

定义于 [src/brain/models.py:73-80](src/brain/models.py#L73-L80)。

表示从 SQLite 读出并经过基础清洗后的评论对象，字段包括：

- `rpid`
- `bvid`
- `uid`
- `message`
- `ctime`

### 4.2 `PatternCard`

定义于 [src/brain/models.py:22-70](src/brain/models.py#L22-L70)。

表示提取出的语言模式卡片，包含：

- `id`
- `title`
- `description`
- `template`
- `examples`
- `frequency`
- `source`
- `created_at`
- `updated_at`

### 4.3 `FrequencyProfile`

定义于 [src/brain/models.py:6-20](src/brain/models.py#L6-L20)。

用于保存模式频率统计，并通过 `freshness` 属性参与检索排序。

## 5. 当前主流程

### 5.1 pipeline 运行流程

`scripts/run_pipeline.py` 的主流程在 [scripts/run_pipeline.py:27-108](scripts/run_pipeline.py#L27-L108)：

1. 读取命令行参数
2. 初始化 `BilibiliReader`、`WatermarkState`、`PatternDB`
3. 列出已完成爬取的视频
4. 按水位线做增量过滤
5. 逐视频读取评论
6. 清洗评论
7. 按块切分评论
8. 对每块调用 LLM 提取模式
9. 将提取结果与已有模式库做去重合并
10. 写入 ChromaDB
11. 更新水位线

### 5.2 查看器流程

`scripts/streamlit_patterns.py` 的主流程在 [scripts/streamlit_patterns.py:50-124](scripts/streamlit_patterns.py#L50-L124)：

1. 初始化 `PatternDB`
2. 读取全部 PatternCard
3. 在“样本浏览”页过滤、排序并展示卡片
4. 在“检索测试”页调用 `retrieve_patterns`
5. 预览 `build_menu` 生成的菜单文本

## 6. 当前实现与旧设计文档的差异

当前实现与旧设计草稿相比，有几处需要特别注意：

- `extract.chunker` 现在只做分块，不做额外清洗，见 [src/brain/extract/chunker.py:5-15](src/brain/extract/chunker.py#L5-L15)
- 评论级去重在 `ingest.cleaner.clean_comments`，不是在 `extract.chunker`
- `extract.refiner.deduplicate_and_merge` 目前只按 `title.strip().lower()` 做去重，不做 embedding 相似度去重，见 [src/brain/extract/refiner.py:83-129](src/brain/extract/refiner.py#L83-L129)
- `viewer` 的实现不在 `src/brain/compose/`，而是在单独脚本 `scripts/streamlit_patterns.py`
- 仓库中没有独立的 `api` 模块

## 7. 推荐阅读顺序

如果你想快速理解当前实现，建议按以下顺序阅读：

1. [01-shared.md](01-shared.md)
2. [02-ingest.md](02-ingest.md)
3. [03-extract.md](03-extract.md)
4. [04-store.md](04-store.md)
5. [05-compose.md](05-compose.md)
6. [06-pipeline.md](06-pipeline.md)
7. [07-viewer.md](07-viewer.md)
