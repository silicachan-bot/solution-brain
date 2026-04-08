# store 模块

`store` 模块负责把 `PatternCard` 持久化到 ChromaDB，并在需要时按语义相关性取回，再结合 `freshness` 做二次排序。

对应代码：
- [src/brain/store/pattern_db.py](../../src/brain/store/pattern_db.py)
- [src/brain/store/embedding.py](../../src/brain/store/embedding.py)
- [src/brain/store/retriever.py](../../src/brain/store/retriever.py)

## 1. 模块职责

`store` 当前负责三件事：

1. 把 `PatternCard` 写入 / 更新到 ChromaDB
2. 提供 embedding function 给 Chroma 查询使用
3. 对检索结果做“相似度 + freshness”排序

它不负责：
- 生成 `PatternCard`
- 修改提示词模板
- UI 展示

## 2. 代码位置

- 持久化层：[src/brain/store/pattern_db.py](../../src/brain/store/pattern_db.py)
- embedding 层：[src/brain/store/embedding.py](../../src/brain/store/embedding.py)
- 检索排序层：[src/brain/store/retriever.py](../../src/brain/store/retriever.py)

## 3. 核心对象 / 函数

### 3.1 `PatternDB`

定义在 [src/brain/store/pattern_db.py:10-76](../../src/brain/store/pattern_db.py#L10-L76)。

它是当前项目对 ChromaDB 的轻量封装。

主要方法：

- `save(cards)`
- `update(cards)`
- `get(pattern_id)`
- `list_all()`
- `query(query_text, top_k=16)`

#### `save(cards)` / `update(cards)`

实现见 [src/brain/store/pattern_db.py:25-35](../../src/brain/store/pattern_db.py#L25-L35)。

行为：
- 空列表直接返回
- 调用 Chroma `upsert`
- `id` 作为向量库主键
- `documents` 使用 `_embed_text(card)` 生成的文本
- `metadatas` 中保存完整 JSON 串

当前 `update(cards)` 只是直接复用 `save(cards)`。

#### `get(pattern_id)`

实现见 [src/brain/store/pattern_db.py:37-45](../../src/brain/store/pattern_db.py#L37-L45)。

行为：
- 按 ID 从 Chroma 取 metadata
- 从 metadata 里的 JSON 恢复 `PatternCard`
- 如果查不到或异常，返回 `None`

#### `list_all()`

实现见 [src/brain/store/pattern_db.py:47-54](../../src/brain/store/pattern_db.py#L47-L54)。

行为：
- 读取整个 collection 的 metadata
- 全量恢复为 `list[PatternCard]`

#### `query(query_text, top_k)`

实现见 [src/brain/store/pattern_db.py:56-71](../../src/brain/store/pattern_db.py#L56-L71)。

行为：
- 若 collection 为空，返回空列表
- 调用 Chroma 的 `query(...)`
- 读取返回的 `distance`
- 用 `1.0 - dist` 转成 similarity
- 返回 `list[(PatternCard, similarity)]`

### 3.2 `_embed_text(card)`

定义在 [src/brain/store/pattern_db.py:73-76](../../src/brain/store/pattern_db.py#L73-L76)。

当前 embedding 文本只由两部分拼成：

```text
description + "例句：" + examples
```

`title` 和 `template` 当前不参与 embedding 文本。

### 3.3 `QwenEmbeddingFunction`

定义在 [src/brain/store/embedding.py:8-18](../../src/brain/store/embedding.py#L8-L18)。

行为：
- 使用 OpenAI 兼容接口
- 读取 embedding 相关配置
- 调用 `client.embeddings.create(...)`
- 返回向量数组

当前默认配置来自 [src/brain/config.py:27-31](../../src/brain/config.py#L27-L31)。

### 3.4 `retrieve_patterns()`

定义在 [src/brain/store/retriever.py:7-22](../../src/brain/store/retriever.py#L7-L22)。

处理流程：

1. 先向量检索 `top_k * 2` 个候选
2. 对每个候选计算综合得分
3. 按综合得分降序排序
4. 取前 `top_k` 条返回

综合得分公式：

```text
score = similarity * SIMILARITY_WEIGHT + freshness * FRESHNESS_WEIGHT
```

权重来自 [src/brain/config.py:37-40](../../src/brain/config.py#L37-L40)。

## 4. 输入与输出

### 4.1 持久化层

输入：
- `list[PatternCard]`
- Chroma 持久化目录

输出：
- 本地 Chroma collection
- 单条或多条 `PatternCard`

### 4.2 embedding 层

输入：
- 文本列表 `Documents`

输出：
- 向量列表 `Embeddings`

### 4.3 检索层

输入：
- `conversation_text`
- `top_k`

输出：
- `list[PatternCard]`

## 5. 执行流程

### 5.1 写入流程

```text
PatternCard
  -> PatternDB._embed_text(card)
  -> Chroma upsert(id, document, metadata.json)
```

### 5.2 查询流程

```text
query_text
  -> Chroma query
  -> 得到 (PatternCard, similarity)
  -> similarity + freshness 重新排序
  -> 返回 top_k PatternCard
```

### 5.3 在项目中的使用位置

- 写入：见 [scripts/run_pipeline.py:90-99](../../scripts/run_pipeline.py#L90-L99)
- 检索：见 [scripts/streamlit_patterns.py:103-120](../../scripts/streamlit_patterns.py#L103-L120)
- 按 ID 读详情：见 [src/brain/compose/tools.py:25-36](../../src/brain/compose/tools.py#L25-L36)

## 6. 依赖关系

依赖：
- `chromadb`
- OpenAI 兼容 embedding API
- [src/brain/models.py](../../src/brain/models.py) 中的 `PatternCard`
- [src/brain/config.py](../../src/brain/config.py) 中的 embedding 配置与检索权重

被这些模块使用：
- [scripts/run_pipeline.py](../../scripts/run_pipeline.py)
- [scripts/streamlit_patterns.py](../../scripts/streamlit_patterns.py)
- [src/brain/compose/tools.py](../../src/brain/compose/tools.py)

## 7. 当前限制 / 已知偏差

1. `query()` 里把相似度简单处理为 `1.0 - distance`。
   - 这依赖当前 collection 的 `cosine` 空间约定，见 [src/brain/store/pattern_db.py:18-20](../../src/brain/store/pattern_db.py#L18-L20)。

2. `PatternCard` 的完整内容放在 metadata JSON 中。
   - 现在实现简单，但 metadata 会随着字段增多而膨胀。

3. embedding 文本当前不包含 `title`。
   - 检索效果主要依赖 `description + examples`。

4. `list_all()` 是全量读取。
   - 适合当前规模，不适合大库分页场景。

5. `PatternDB` 没有单独抽象 repository 接口。
   - 目前直接把 Chroma 当成主存储和检索层。

## 8. 测试覆盖

测试位于 [tests/test_store.py](../../tests/test_store.py)。

覆盖点包括：
- `save()` / `get()`
- `list_all()`
- `update()`
- `retrieve_patterns()` 的空库与 `top_k`
- `QwenEmbeddingFunction` 的 SDK 调用包装
