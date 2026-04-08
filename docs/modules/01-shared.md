# shared 模块说明

`shared` 不是一个单独目录，而是当前仓库里由 `models.py` 和 `config.py` 组成的一组公共基础定义，供其他模块统一引用。

## 1. 模块职责

`shared` 负责两件事：

1. 定义跨模块共享的数据结构
2. 定义运行时配置与默认路径

对应代码：

- [src/brain/models.py](src/brain/models.py)
- [src/brain/config.py](src/brain/config.py)

## 2. 代码位置

### 2.1 数据模型

- [src/brain/models.py:6-20](src/brain/models.py#L6-L20) `FrequencyProfile`
- [src/brain/models.py:22-70](src/brain/models.py#L22-L70) `PatternCard`
- [src/brain/models.py:73-80](src/brain/models.py#L73-L80) `CleanedComment`

### 2.2 配置

- [src/brain/config.py:7-14](src/brain/config.py#L7-L14) 路径与数据目录
- [src/brain/config.py:15-20](src/brain/config.py#L15-L20) Bilibili 数据库路径
- [src/brain/config.py:22-31](src/brain/config.py#L22-L31) LLM 与 embedding 配置
- [src/brain/config.py:33-40](src/brain/config.py#L33-L40) 提取与检索参数

## 3. 核心对象

### 3.1 `FrequencyProfile`

用于记录模式的频率统计：

- `recent`
- `medium`
- `long_term`
- `total`

并提供 `freshness` 属性：

- 计算逻辑见 [src/brain/models.py:13-20](src/brain/models.py#L13-L20)
- 当前实现使用 `recent_ratio * 0.35 + medium_ratio * 0.65`
- `long_term` 与 `total` 参与数据保存，但当前 `freshness` 计算不直接使用 `long_term`

### 3.2 `PatternCard`

`PatternCard` 是整个项目的核心结构，字段见 [src/brain/models.py:22-33](src/brain/models.py#L22-L33)。

它还提供：

- `to_dict()`：序列化为 JSON 友好的字典，见 [src/brain/models.py:34-50](src/brain/models.py#L34-L50)
- `from_dict()`：从存储字典恢复对象，见 [src/brain/models.py:52-70](src/brain/models.py#L52-L70)

该对象在以下位置被直接使用：

- `extract.refiner` 生成新卡片
- `store.pattern_db` 存储/读取卡片
- `store.retriever` 返回检索结果
- `compose` 构造菜单和工具返回值
- `viewer` 展示卡片详情

### 3.3 `CleanedComment`

定义于 [src/brain/models.py:73-80](src/brain/models.py#L73-L80)。

该对象表示已经被 ingest 模块接受的评论。当前字段较少，只保留后续提取阶段需要的最小信息。

## 4. 输入与输出

### 4.1 `models.py`

- 输入：无外部输入
- 输出：供其他模块 import 的 dataclass 定义

### 4.2 `config.py`

- 输入：`.env` 和系统环境变量，见 [src/brain/config.py:8-9](src/brain/config.py#L8-L9)
- 输出：一组模块级常量

主要包括：

- `SOLUTION_BRAIN_DIR`
- `DATA_DIR`
- `STATE_FILE`
- `CHROMA_DIR`
- `BILIBILI_DB_PATH`
- `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL`
- `EMBED_API_BASE` / `EMBED_API_KEY` / `EMBED_MODEL` / `EMBED_DIMENSIONS`
- `CHUNK_SIZE`
- `MIN_COMMENT_LENGTH`
- `RETRIEVAL_TOP_K`
- `SIMILARITY_WEIGHT`
- `FRESHNESS_WEIGHT`

## 5. 执行流程

### 5.1 配置加载流程

`config.py` 在 import 时立即执行以下步骤：

1. 计算仓库根目录 `SOLUTION_BRAIN_DIR`
2. 调用 `load_dotenv(SOLUTION_BRAIN_DIR / ".env")`
3. 从环境变量读取配置；如果缺失则回退到默认值

这意味着：

- 各模块不需要显式传入这些默认配置
- 但 import 时就会固定当前环境值

### 5.2 数据模型流转

```text
SQLite comment row
    ↓
CleanedComment
    ↓
extract 产出
    ↓
PatternCard
    ↓
PatternCard.to_dict()
    ↓
JSON metadata in ChromaDB
    ↓
PatternCard.from_dict()
    ↓
compose / viewer / retriever 消费
```

## 6. 依赖关系

### 6.1 谁依赖 `models.py`

几乎所有模块都依赖：

- `ingest.reader`
- `ingest.cleaner`
- `extract.chunker`
- `extract.refiner`
- `store.pattern_db`
- `store.retriever`
- `compose.assembler`
- `compose.menu`
- `viewer`

### 6.2 谁依赖 `config.py`

主要包括：

- `ingest.cleaner`：`MIN_COMMENT_LENGTH`
- `extract.refiner`：LLM 配置
- `store.embedding`：embedding 配置
- `store.retriever`：检索排序权重
- `scripts/run_pipeline.py`：路径与默认参数
- `scripts/streamlit_patterns.py`：Chroma 路径与默认 top_k

## 7. 当前限制 / 已知偏差

1. `shared` 不是独立包，只是按语义归类的公共文件
2. `config.py` 使用模块级常量，没有分环境对象或显式配置注入
3. `BILIBILI_DB_PATH` 的默认值依赖 monorepo 相对路径，见 [src/brain/config.py:15-20](src/brain/config.py#L15-L20)
4. `PatternCard` 的 schema 目前稳定，但没有版本字段
5. `FrequencyProfile.freshness` 的计算是当前实现策略，不代表最终产品定义
