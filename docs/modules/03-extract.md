# extract 模块

`extract` 模块负责把清洗后的评论变成 `PatternCard`。当前实现分成两步：按块切分评论，再逐块调用 LLM 提取模式，最后做模式级去重合并。

对应代码：
- [src/brain/extract/chunker.py](../../src/brain/extract/chunker.py)
- [src/brain/extract/refiner.py](../../src/brain/extract/refiner.py)
- [src/brain/prompts/extract_patterns.txt](../../src/brain/prompts/extract_patterns.txt)
- [src/brain/prompts/extract_dedup_judge.txt](../../src/brain/prompts/extract_dedup_judge.txt)

## 1. 模块职责

`extract` 当前负责：

1. 将单视频评论切成固定大小的块
2. 对每一块评论调用 LLM
3. 把 LLM JSON 输出转成 `PatternCard`
4. 将新模式彼此合并，并和已有模式库合并

它不负责：
- 评论清洗
- embedding 生成
- 向量存储
- system prompt 组装

## 2. 代码位置

- 分块器：[src/brain/extract/chunker.py](../../src/brain/extract/chunker.py)
- 提取与合并：[src/brain/extract/refiner.py](../../src/brain/extract/refiner.py)
- [src/brain/prompts/extract_patterns.txt](../../src/brain/prompts/extract_patterns.txt)
- [src/brain/prompts/extract_dedup_judge.txt](../../src/brain/prompts/extract_dedup_judge.txt)

prompt 文件统一位于 `src/brain/prompts/`，命名规范为 `模块名_用途.txt`。

## 3. 核心对象 / 函数

### 3.1 `chunk_comments()`

定义在 [src/brain/extract/chunker.py:5-15](../../src/brain/extract/chunker.py#L5-L15)。

行为非常直接：
- 输入 `list[CleanedComment]`
- 取出每条的 `message`
- 按 `chunk_size` 切成 `list[list[str]]`

当前它只做切片，不做额外清洗或去重。

### 3.2 `extract_patterns.txt`

位于 [src/brain/prompts/extract_patterns.txt](../../src/brain/prompts/extract_patterns.txt)。

这是当前用于模式提取的提示词模板，要求模型输出 JSON 数组，每个元素包含：
- `template`
- `examples`
- `description`

评论正文通过 `{{ comments }}` 注入模板。

### 3.3 `_client` 与 `_call_llm_streaming()`

`_client` 是模块级 `OpenAI` 单例，避免每次调用重建连接池。

`_call_llm_streaming()` 使用流式接口：
- `stream=True` + `stream_options={"include_usage": True}`
- 逐 token 调用可选的 `on_token(n)` 回调（`n` 为当前 completion token 累计数）
- 最后一个 chunk 含 usage 时读取 `prompt_tokens` / `completion_tokens`
- 返回 `(content: str, prompt_tokens: int, completion_tokens: int)`

`_llm_logger` 为模块级文件日志器，把每次调用的完整 prompt 和模型原始回复写入 `data/llm_responses.log`，不打印到终端。

### 3.4 `extract_from_chunk()`

签名：`extract_from_chunk(messages, log_label="", on_token=None) -> (list[PatternCard], int)`

`on_token` 是可选回调，每生成一个 token 触发一次，用于调用方实时更新进度显示。

处理过程：

1. 给评论编号
2. 渲染 [src/brain/prompts/extract_patterns.txt](../../src/brain/prompts/extract_patterns.txt)
3. 流式调用 LLM，边生成边触发 `on_token`
4. 将完整 prompt 与模型回复写入后台日志
5. 如有 Markdown code fence，则剥掉
6. `json.loads(...)` 解析
7. 过滤掉缺字段的项
8. 为每个有效结果构造 `PatternCard`

返回值为 `(cards, total_tokens)`，`total_tokens=0` 表示 API 未返回用量。

构造 `PatternCard` 时的当前约定：
- `id`：随机 `pat-xxxxxxxx`
- `examples`：最多保留前 5 条
- `frequency`：初始四个窗口都设为 1
- `source`：固定写成 `bilibili`
- 校验必要字段为 `template`、`examples`、`description`，缺任何一个则丢弃该条目

如果返回不是合法 JSON 或不是数组，返回 `([], total_tokens)`。

### 3.5 `deduplicate_and_merge()`

定义在 [src/brain/extract/refiner.py:378-399](../../src/brain/extract/refiner.py#L378-L399)。

签名：`deduplicate_and_merge(cards, db, embedder, top_n=DEDUP_TOP_N)`

这是两阶段去重的统一入口：

1. 先调用 `_dedup_intra_batch(cards, embedder, top_n)` 完成批次内去重
2. 再调用 `_dedup_against_db(deduped, db, top_n)` 完成入库前去重

返回值：
- `new_cards`：需要新增到数据库的卡片
- `updated_existing_cards`：需要更新已有记录的卡片

### 3.6 `_judge_duplicate_topn(card, candidates)`

定义在 [src/brain/extract/refiner.py:117-159](../../src/brain/extract/refiner.py#L117-L159)。

把当前模式和 top-N 候选渲染进 [src/brain/prompts/extract_dedup_judge.txt](../../src/brain/prompts/extract_dedup_judge.txt)，再一次性交给 LLM，判断候选中是否有与当前模式语义等价或高度相似的条目。

返回 `(candidate_index_0based | None, 'current' | 'candidate')`：
- 第一项为 `None` 表示无重复；为整数时表示 `candidates` 列表中匹配的 0-based 下标
- 第二项表示哪一方的描述更完整，仅在有重复时有意义

解析失败或候选为空时安全返回 `(None, 'current')`。

### 3.7 `_dedup_intra_batch(cards, embedder, top_n)`

定义在 [src/brain/extract/refiner.py:240-325](../../src/brain/extract/refiner.py#L240-L325)。

批次内去重，核心逻辑：

1. 用 `tempfile.TemporaryDirectory` 创建临时 LanceDB 表，schema 含 `id`、`json`、`vec_template`、`vec_semantic` 四列
2. 第一张卡片直接插入，不做检索
3. 后续每张卡片先对 `vec_template` 和 `vec_semantic` 各取 top_n 候选
4. 调用 `_merge_hits` 合并去重后交 `_judge_duplicate_topn` 判断
5. 判断重复则合并到已有卡片，更新临时表；不重复则插入临时表并保留

终端使用 `rich.Console` 并列输出两路检索结果和 LLM 判断结果。

返回批次内去重后的 `list[PatternCard]`。

### 3.8 `_dedup_against_db(cards, db, top_n)`

定义在 [src/brain/extract/refiner.py:328-375](../../src/brain/extract/refiner.py#L328-L375)。

入库前去重，核心逻辑：

1. 每张新卡片对持久库分别调用 `db.query_by_template(card.template, top_k=top_n)` 和 `db.query_by_semantic(card.embed_text(), top_k=top_n)`
2. 调用 `_merge_hits` 合并后交 `_judge_duplicate_topn` 判断
3. 若判断重复，累计到 `updates_by_id` 字典；多张新卡片匹配同一已有卡片时，频率合并到同一对象上
4. 不重复则加入 `new_cards`

终端并列输出两路检索结果和 LLM 判断结果。

返回 `(new_cards, list(updates_by_id.values()))`。

### 3.9 `_merge_hits(hits_a, hits_b)`

定义在 [src/brain/extract/refiner.py:228-237](../../src/brain/extract/refiner.py#L228-L237)。

合并两路检索结果（均为 `list[tuple[PatternCard, float]]`），按 `card.id` 去重后返回 `list[PatternCard]`。

### 3.10 `_merge_into()`

定义在 [src/brain/extract/refiner.py:402-413](../../src/brain/extract/refiner.py#L402-L413)。

合并规则：
- `frequency` 四个字段逐项累加
- `examples` 去重后补充，最多保留 5 条
- `updated_at` 刷新为当前时间

## 4. 输入与输出

### 4.1 分块阶段

输入：
- `list[CleanedComment]`

输出：
- `list[list[str]]`

### 4.2 提取阶段

输入：
- 单个评论块 `list[str]`
- 可选 `log_label: str`
- 可选 `on_token: Callable[[int], None]`

输出：
- `(list[PatternCard], total_tokens: int)`

### 4.3 合并阶段

输入：
- 新提取的 `list[PatternCard]`（`cards`）
- 持久库对象 `db`（`PatternDB`）
- embedding 对象 `embedder`（`QwenEmbedder`）
- 可选 `top_n: int`

输出：
- `(new_cards, updates)`

## 5. 执行流程

当前 `extract` 在 pipeline 中的实际流程：

```text
cleaned comments
  -> chunk_comments(...)
  -> for each chunk: extract_from_chunk(...)
  -> all_patterns
  -> deduplicate_and_merge(all_patterns, db, embedder)
```

展开后是：

```text
1. 一个视频的评论切成多块
2. 每块独立请求一次 LLM
3. 每块返回若干 PatternCard
4. 汇总所有块的 PatternCard
5. 阶段一：_dedup_intra_batch —— 批次内双路向量检索 + LLM 判重，合并同批次重复模式
6. 阶段二：_dedup_against_db —— 与持久库双路向量检索 + LLM 判重，区分新增与更新
7. 返回 new_cards 和 updates
```

## 6. 依赖关系

依赖：
- [src/brain/models.py](../../src/brain/models.py) 中的 `CleanedComment`、`PatternCard`、`FrequencyProfile`
- [src/brain/config.py](../../src/brain/config.py) 中的 LLM 配置、`DEDUP_TOP_N`、`EMBED_DIMENSIONS`
- OpenAI 兼容 SDK
- `lancedb`、`pyarrow`（批次内临时表）
- `rich.console.Console`（终端日志）

被这些模块使用：
- [scripts/run_pipeline.py](../../scripts/run_pipeline.py)
- 测试文件 [tests/test_extract.py](../../tests/test_extract.py)

## 7. 当前限制 / 已知偏差

1. `chunk_comments()` 当前**不做**设计文档里提过的基础清洗去重。
   - 评论清洗实际发生在 `ingest.cleaner`，而不是 `extract.chunker`。

2. 模式去重使用双路向量检索 + LLM 判重，每张卡片都会产生 LLM 调用开销。
   - 批次较大时，去重阶段的 API 成本不可忽视。

3. `extract_from_chunk()` 对异常 JSON 的处理很保守。
   - 当前解析失败就返回空列表，不保留原始错误上下文。

4. `source` 当前固定写死为 `bilibili`。
   - 这意味着 `PatternCard` 还没有携带更细粒度的数据来源信息。

5. 当前没有”跨块二次验证”或”模式质量评分”。
   - LLM 提什么就收什么，入库前只做向量 + LLM 去重，不评估质量。

6. `_dedup_intra_batch` 使用 `tempfile.TemporaryDirectory`，临时 LanceDB 表在函数返回后立即删除。
   - 批次内去重无法持久化中间状态，失败时需重跑整批。
