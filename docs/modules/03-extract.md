# extract 模块

`extract` 模块负责把清洗后的评论整理成“评论对”，再逐块调用 LLM 提取 `PatternCard`，最后做模式级去重合并。

对应代码：
- [src/brain/extract/chunker.py](../../src/brain/extract/chunker.py)
- [src/brain/extract/refiner.py](../../src/brain/extract/refiner.py)
- [src/brain/prompts/extract_patterns.txt](../../src/brain/prompts/extract_patterns.txt)
- [src/brain/prompts/extract_dedup_judge.txt](../../src/brain/prompts/extract_dedup_judge.txt)

## 1. 模块职责

`extract` 当前负责：

1. 从清洗后的评论中构造评论对
2. 将评论对切成固定大小的块
3. 对每一块评论对调用 LLM
4. 把 LLM JSON 输出转成 `PatternCard`
5. 将新模式彼此合并，并和已有模式库合并

它不负责：
- 评论清洗
- embedding 生成
- 向量存储
- system prompt 组装

## 2. 核心对象 / 函数

### 2.1 `build_comment_pairs()`

定义在 [src/brain/extract/chunker.py](../../src/brain/extract/chunker.py)。

行为：
- 输入 `list[CleanedComment]`
- 对所有 `root/parent` 非 0 的评论，找到它对应的上文评论
- 输出 `list[CommentPair]`

当前规则：
- 只保留“上文评论 + 回复评论”的成对输入
- 顶层未被回复的评论不会进入提取
- 优先使用 `parent` 找直接上文；缺失时退回 `root`

### 2.2 `chunk_comments()`

输入 `list[CommentPair]`，先把每个 pair 格式化为：

```text
上文评论：...
回复评论：...
```

然后按 `chunk_size` 切成 `list[list[str]]`。

### 2.3 `extract_patterns.txt`

位于 [src/brain/prompts/extract_patterns.txt](../../src/brain/prompts/extract_patterns.txt)。

这是当前用于模式提取的提示词模板。当前要求模型：
- 只从“回复评论”里提取模式
- 只保留适合对话回复复用的表达
- 宁缺毋滥，一批最多输出 3 个高质量模式

输出字段仍为：
- `template`
- `examples`
- `description`

### 2.4 `extract_from_chunk()`

签名：`extract_from_chunk(messages, log_label="", on_token=None) -> (list[PatternCard], int)`

处理过程：
1. 给评论对编号
2. 渲染 [src/brain/prompts/extract_patterns.txt](../../src/brain/prompts/extract_patterns.txt)
3. 流式调用 LLM
4. 记录完整 prompt 和模型回复
5. 解析 JSON
6. 转成 `PatternCard`

### 2.5 `deduplicate_and_merge()`

签名：`deduplicate_and_merge(cards, db, embedder, top_n=DEDUP_TOP_N)`

分两阶段：
1. `_dedup_intra_batch()` 做批次内去重
2. `_dedup_against_db()` 做入库前去重

## 3. 输入与输出

### 3.1 评论对构造阶段

输入：
- `list[CleanedComment]`

输出：
- `list[CommentPair]`

### 3.2 分块阶段

输入：
- `list[CommentPair]`

输出：
- `list[list[str]]`

### 3.3 提取阶段

输入：
- 单个评论对块 `list[str]`
- 可选 `log_label: str`
- 可选 `on_token: Callable[[int], None]`

输出：
- `(list[PatternCard], total_tokens: int)`

### 3.4 合并阶段

输入：
- 新提取的 `list[PatternCard]`
- `PatternDB`
- `QwenEmbedder`

输出：
- `(new_cards, updates)`

## 4. 执行流程

```text
cleaned comments
  -> build_comment_pairs(...)
  -> chunk_comments(...)
  -> for each chunk: extract_from_chunk(...)
  -> all_patterns
  -> deduplicate_and_merge(all_patterns, db, embedder)
```

## 5. 依赖关系

依赖：
- [src/brain/models.py](../../src/brain/models.py) 中的 `CleanedComment`、`CommentPair`、`PatternCard`
- [src/brain/config.py](../../src/brain/config.py) 中的 LLM 配置、`DEDUP_TOP_N`、`EMBED_DIMENSIONS`
- OpenAI 兼容 SDK
- `lancedb`、`pyarrow`

被这些模块使用：
- [scripts/run_pipeline.py](../../scripts/run_pipeline.py)
- [tests/test_extract.py](../../tests/test_extract.py)

## 6. 当前限制 / 已知偏差

1. 当前只提取“评论对”中的回复评论，顶层未被回复的评论默认不会进入提取。
2. `build_comment_pairs()` 只能使用当前列表里能找到上文的回复；如果上文缺失，该回复会被跳过。
3. 模式去重仍使用双路向量检索 + LLM 判重，批次较大时成本不可忽视。
4. `extract_from_chunk()` 对异常 JSON 的处理仍然偏保守，解析失败直接返回空列表。
5. 当前没有跨块质量评分，只靠 prompt 约束 + 入库前去重控制结果质量。
