# extract 模块

`extract` 模块负责把清洗后的评论变成 `PatternCard`。当前实现分成两步：按块切分评论，再逐块调用 LLM 提取模式，最后做模式级去重合并。

对应代码：
- [src/brain/extract/chunker.py](../../src/brain/extract/chunker.py)
- [src/brain/extract/refiner.py](../../src/brain/extract/refiner.py)

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

## 3. 核心对象 / 函数

### 3.1 `chunk_comments()`

定义在 [src/brain/extract/chunker.py:5-15](../../src/brain/extract/chunker.py#L5-L15)。

行为非常直接：
- 输入 `list[CleanedComment]`
- 取出每条的 `message`
- 按 `chunk_size` 切成 `list[list[str]]`

当前它只做切片，不做额外清洗或去重。

### 3.2 `_EXTRACT_PROMPT`

定义在 [src/brain/extract/refiner.py:11-29](../../src/brain/extract/refiner.py#L11-L29)。

这是当前用于模式提取的固定提示词，要求模型输出 JSON 数组，每个元素包含：
- `title`
- `template`
- `examples`
- `description`

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
2. 拼接提示词
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

如果返回不是合法 JSON 或不是数组，返回 `([], total_tokens)`。

### 3.5 `deduplicate_and_merge()`

定义在 [src/brain/extract/refiner.py:83-115](../../src/brain/extract/refiner.py#L83-L115)。

当前去重逻辑是 MVP 版本：**按标题归一化后精确匹配**。

归一化方式：

```text
card.title.strip().lower()
```

它做两轮合并：

1. 新卡片之间合并
2. 新卡片与已有模式库合并

返回值：
- `new_cards`
- `updated_existing_cards`

### 3.6 `_merge_into()`

定义在 [src/brain/extract/refiner.py:118-129](../../src/brain/extract/refiner.py#L118-L129)。

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
- 新提取的 `list[PatternCard]`
- 已存在的 `list[PatternCard]`

输出：
- `(new_cards, updates)`

## 5. 执行流程

当前 `extract` 在 pipeline 中的实际流程：

```text
cleaned comments
  -> chunk_comments(...)
  -> for each chunk: extract_from_chunk(...)
  -> all_patterns
  -> deduplicate_and_merge(all_patterns, existing_patterns)
```

展开后是：

```text
1. 一个视频的评论切成多块
2. 每块独立请求一次 LLM
3. 每块返回若干 PatternCard
4. 汇总所有块的 PatternCard
5. 按标题做去重和频率合并
6. 区分出 new_cards 和 updates
```

## 6. 依赖关系

依赖：
- [src/brain/models.py](../../src/brain/models.py) 中的 `CleanedComment`、`PatternCard`、`FrequencyProfile`
- [src/brain/config.py](../../src/brain/config.py) 中的 LLM 配置
- OpenAI 兼容 SDK

被这些模块使用：
- [scripts/run_pipeline.py](../../scripts/run_pipeline.py)
- 测试文件 [tests/test_extract.py](../../tests/test_extract.py)

## 7. 当前限制 / 已知偏差

1. `chunk_comments()` 当前**不做**设计文档里提过的基础清洗去重。
   - 评论清洗实际发生在 `ingest.cleaner`，而不是 `extract.chunker`。

2. 模式去重目前只按标题精确匹配。
   - 例如标题不同但语义相近的模式，不会合并。

3. `extract_from_chunk()` 对异常 JSON 的处理很保守。
   - 当前解析失败就返回空列表，不保留原始错误上下文。

4. `source` 当前固定写死为 `bilibili`。
   - 这意味着 `PatternCard` 还没有携带更细粒度的数据来源信息。

5. 当前没有“跨块二次验证”或“模式质量评分”。
   - LLM 提什么就收什么，后续只做标题级 merge。
