# store 模块

`store` 模块负责把 `PatternCard` 持久化到 LanceDB，并在需要时按语义相关性取回，再结合 `freshness` 做二次排序。

对应代码：
- [src/brain/store/pattern_db.py](../../src/brain/store/pattern_db.py)
- [src/brain/store/embedding.py](../../src/brain/store/embedding.py)
- [src/brain/store/retriever.py](../../src/brain/store/retriever.py)

## 1. 模块职责

`store` 当前负责三件事：

1. 把 `PatternCard` 写入 / 更新到 LanceDB，同时维护**双向量列**
2. 提供 embedding 接口 `QwenEmbedder` 供写入和查询使用
3. 对检索结果做"相似度 + freshness"排序

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

定义在 [src/brain/store/pattern_db.py:13-126](../../src/brain/store/pattern_db.py#L13-L126)。

它是当前项目对 LanceDB 的轻量封装，持久化目录由 `config.py` 中的 `LANCEDB_DIR` 提供。

**构造**：

```python
PatternDB(persist_dir, embedder=None)
```

- `lancedb.connect(str(persist_dir))` 连接（或创建）本地 LanceDB 数据库
- 尝试打开已有的 `patterns` 表；若不存在则将 `self._table` 置为 `None`
- `embedder` 只在写入和查询时使用，`get()` / `list_all()` / `count()` 不需要它

**表结构**：

由 [src/brain/store/pattern_db.py:26-32](../../src/brain/store/pattern_db.py#L26-L32) 的 `_make_schema()` 定义，共四列：

| 列名 | 类型 | 说明 |
|---|---|---|
| `id` | `string` | PatternCard 主键 |
| `json` | `string` | PatternCard 完整 JSON 序列化 |
| `vec_template` | `list<float32>[EMBED_DIMENSIONS]` | template 句式的 embedding |
| `vec_semantic` | `list<float32>[EMBED_DIMENSIONS]` | description + examples 的 embedding |

主要方法：

- `save(cards)` / `update(cards)`
- `get(pattern_id)`
- `list_all()`
- `count()`
- `query_by_template(text, top_k)`
- `query_by_semantic(text, top_k)`

#### `save(cards)` / `update(cards)`

实现见 [src/brain/store/pattern_db.py:49-64](../../src/brain/store/pattern_db.py#L49-L64)。

行为：
- 空列表直接返回
- 调用 `_cards_to_data(cards)` 生成双向量数据
- 若表不存在，调用 `db.create_table(...)` 首次建表
- 若表已存在，调用 `merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(data)` 实现 upsert

`update(cards)` 当前直接复用 `save(cards)`。

#### `get(pattern_id)`

实现见 [src/brain/store/pattern_db.py:66-78](../../src/brain/store/pattern_db.py#L66-L78)。

行为：
- 若表不存在，返回 `None`
- 使用 `.search().where(f"id = '{pattern_id}'").limit(1)` 过滤
- 从结果的 `json` 列反序列化为 `PatternCard`
- 查不到或发生异常均返回 `None`

不需要 embedder。

#### `list_all()`

实现见 [src/brain/store/pattern_db.py:80-87](../../src/brain/store/pattern_db.py#L80-L87)。

行为：
- 若表不存在，返回空列表
- 调用 `to_pandas()` 全量读取
- 逐行从 `json` 列恢复 `PatternCard`

不需要 embedder。

#### `count()`

实现见 [src/brain/store/pattern_db.py:89-92](../../src/brain/store/pattern_db.py#L89-L92)。

行为：
- 若表不存在，返回 `0`
- 调用 `table.count_rows()` 返回记录总数

不需要 embedder。

#### `query_by_template(text, top_k=3)`

实现见 [src/brain/store/pattern_db.py:94-109](../../src/brain/store/pattern_db.py#L94-L109)。

行为：
- 若表为空，返回空列表
- 对 `text` 调用 embedder 得到向量
- 使用 `vector_column_name="vec_template"` 在 template 向量列上做 cosine 检索
- 将 LanceDB 返回的 `_distance` 转成 similarity（`1.0 - distance`）
- 返回 `list[tuple[PatternCard, float]]`

**用途**：去重阶段，按句式结构相似性找同类模式。

#### `query_by_semantic(text, top_k=8)`

实现见 [src/brain/store/pattern_db.py:111-126](../../src/brain/store/pattern_db.py#L111-L126)。

行为：
- 与 `query_by_template` 流程相同
- 使用 `vector_column_name="vec_semantic"` 在 semantic 向量列上做 cosine 检索
- 返回 `list[tuple[PatternCard, float]]`

**用途**：对话检索阶段，按语义内容（含义、例句）找相关模式。

### 3.2 `_cards_to_data(cards)`

定义在 [src/brain/store/pattern_db.py:34-47](../../src/brain/store/pattern_db.py#L34-L47)。

这是生成双向量的核心辅助函数：

1. 提取所有卡片的 `card.template` 字符串列表
2. 提取所有卡片的 `card.embed_text()` 字符串列表
3. 分别批量调用 `embedder.embed(...)` 得到 `vec_t` 和 `vec_s`
4. 将 id、json、vec_template、vec_semantic 组合成 dict 列表返回

`PatternCard.embed_text()` 定义在 [src/brain/models.py:70-73](../../src/brain/models.py#L70-L73)，返回：

```text
"{description} 例句：{example1} / {example2} / ..."
```

### 3.3 两个向量列的用途区分

这是当前存储层的核心设计：每张 `PatternCard` 在写入时同时生成两条独立的向量。

| 向量列 | embedding 来源 | 检索用途 |
|---|---|---|
| `vec_template` | `card.template`（句式模板） | 去重阶段：检索结构相近的同类模式 |
| `vec_semantic` | `card.embed_text()`（description + examples） | 对话检索：按语境相关性找适合使用的模式 |

两者对应两个不同的使用场景：
- 去重场景关心的是"这个句式是否已经存在了"，所以用 template 向量
- 检索场景关心的是"当前对话情境下哪些模式最合适"，所以用 semantic 向量

### 3.4 `QwenEmbedder`

定义在 [src/brain/store/embedding.py:8-22](../../src/brain/store/embedding.py#L8-L22)。

替代旧的 `QwenEmbeddingFunction`，提供更简洁的接口：

```python
embedder.embed(texts: list[str]) -> list[list[float]]
```

行为：
- 空列表直接返回 `[]`，不发起 API 请求
- 使用 OpenAI 兼容 SDK 调用 embedding 接口
- 读取配置来自 [src/brain/config.py:29-32](../../src/brain/config.py#L29-L32)：`EMBED_API_BASE`、`EMBED_API_KEY`、`EMBED_MODEL`、`EMBED_DIMENSIONS`
- 返回每条文本对应的向量，维度为 `EMBED_DIMENSIONS`（默认 2048）

### 3.5 `retrieve_patterns()`

定义在 [src/brain/store/retriever.py:8-23](../../src/brain/store/retriever.py#L8-L23)。

处理流程：

1. 调用 `db.query_by_semantic(conversation_text, top_k=top_k * 2)` 先取双倍候选
2. 对每个候选计算综合得分
3. 按综合得分降序排序
4. 取前 `top_k` 条返回

综合得分公式：

```text
score = similarity * SIMILARITY_WEIGHT + freshness * FRESHNESS_WEIGHT
```

权重来自 [src/brain/config.py:40-41](../../src/brain/config.py#L40-L41)：

- `SIMILARITY_WEIGHT = 0.6`
- `FRESHNESS_WEIGHT = 0.4`

## 4. 输入与输出

### 4.1 持久化层

输入：
- `list[PatternCard]`
- LanceDB 持久化目录（`LANCEDB_DIR`）
- `QwenEmbedder` 实例

输出：
- 本地 LanceDB 表（`patterns`）
- 单条或多条 `PatternCard`

### 4.2 embedding 层

输入：
- 文本列表 `list[str]`

输出：
- 向量列表 `list[list[float]]`

### 4.3 检索层

输入：
- `conversation_text`
- `top_k`

输出：
- `list[PatternCard]`

## 5. 执行流程

### 5.1 写入流程

```text
PatternCard 列表
  -> _cards_to_data(cards)
     -> embedder.embed(templates)  -> vec_template 列表
     -> embedder.embed(embed_texts) -> vec_semantic 列表
     -> 组装 {id, json, vec_template, vec_semantic} 列表
  -> 首次：create_table(...)
     已有：merge_insert("id").when_matched_update_all().when_not_matched_insert_all()
```

### 5.2 查询流程（semantic 检索）

```text
conversation_text
  -> embedder.embed([conversation_text])[0]
  -> table.search(vec, vector_column_name="vec_semantic").metric("cosine")
  -> 得到 (PatternCard, similarity) 列表
  -> similarity * 0.6 + freshness * 0.4 重新排序
  -> 返回前 top_k 条 PatternCard
```

### 5.3 查询流程（template 检索，用于去重）

```text
template_text
  -> embedder.embed([template_text])[0]
  -> table.search(vec, vector_column_name="vec_template").metric("cosine")
  -> 返回 (PatternCard, similarity) 列表
```

### 5.4 在项目中的使用位置

- 写入：见 [scripts/run_pipeline.py](../../scripts/run_pipeline.py)
- 检索：见 [scripts/streamlit_patterns.py](../../scripts/streamlit_patterns.py)
- 按 ID 读详情：见 [src/brain/compose/tools.py](../../src/brain/compose/tools.py)

## 6. 依赖关系

依赖：
- `lancedb`
- `pyarrow`
- OpenAI 兼容 embedding API
- [src/brain/models.py](../../src/brain/models.py) 中的 `PatternCard`
- [src/brain/config.py](../../src/brain/config.py) 中的 embedding 配置与检索权重

被这些模块使用：
- [scripts/run_pipeline.py](../../scripts/run_pipeline.py)
- [scripts/streamlit_patterns.py](../../scripts/streamlit_patterns.py)
- [src/brain/compose/tools.py](../../src/brain/compose/tools.py)

## 7. 当前限制 / 已知偏差

1. `query_by_template()` 和 `query_by_semantic()` 均把相似度处理为 `1.0 - distance`。
   - 这依赖两个向量列都使用 `cosine` 度量，见 [src/brain/store/pattern_db.py:103](../../src/brain/store/pattern_db.py#L103) 和 [src/brain/store/pattern_db.py:120](../../src/brain/store/pattern_db.py#L120)。

2. `PatternCard` 的完整内容序列化后存在 `json` 列。
   - 实现简单，但字段增多时 JSON 列会膨胀；不支持对内部字段直接过滤。

3. `list_all()` 是全量读取，通过 `to_pandas()` 一次性载入。
   - 适合当前规模，不适合大库分页场景。

4. 写入和查询都需要 embedder，但 `get()` / `list_all()` / `count()` 不需要。
   - 构造时 `embedder=None` 是合法的，但调用写入或查询方法时会抛异常。

5. `PatternDB` 没有单独抽象 repository 接口。
   - 目前直接把 LanceDB 当成主存储和检索层。

6. 双向量写入每次都要发起两次 embedding API 批量请求。
   - 批量调用已经合并到一次 `embed(templates)` 和一次 `embed(semantics)`，但对大批量写入仍有 API 延迟。

## 8. 测试覆盖

测试位于 [tests/test_store.py](../../tests/test_store.py)。

覆盖点包括：
- `save()` / `get()`：写入后按 ID 读取，验证字段一致
- `list_all()`：多条写入后全量读取数量
- `update()`：覆盖写入后字段已更新
- `count()`：空库返回 0，写入后返回正确数量
- `test_query_by_template`：按 template 向量列检索，验证返回数量与 similarity 类型
- `test_query_by_semantic`：按 semantic 向量列检索，验证返回数量
- 空库查询：`query_by_template` 和 `query_by_semantic` 均返回空列表
- `retrieve_patterns()` 的空库与 `top_k` 限制
- `QwenEmbedder` 的 SDK 调用包装与空列表短路
