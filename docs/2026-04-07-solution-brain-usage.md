# solution-brain MVP 使用说明

## 1. 当前已经做了什么

本次 MVP 已完成以下模块：

- **ingest**：从 Bilibili SQLite 数据库读取评论，做基础清洗和去重
- **extract**：按块调用 LLM 提取语言模式，并做跨块/跨视频去重合并
- **store**：将 PatternCard 存入 ChromaDB，并支持按相似度 + freshness 检索
- **compose**：把检索到的模式组装成渐进式菜单，并提供 `inspect_pattern` 工具定义
- **pipeline**：提供命令行脚本，把 ingest → extract → store 串起来跑通

## 2. 当前可运行的脚本

目前仓库里可直接运行的主脚本有两个：

### `scripts/run_pipeline.py`

用途：运行完整提取 pipeline。

常用命令：

```bash
uv run python scripts/run_pipeline.py
```

增量处理（默认行为，只处理水位线之后的新视频）。

```bash
uv run python scripts/run_pipeline.py --full
```

全量重跑，忽略水位线。

```bash
uv run python scripts/run_pipeline.py --max-chunks 5
```

每个视频最多处理前 5 块评论对，适合控制成本做小规模验证。

```bash
uv run python scripts/run_pipeline.py --chunk-size 30
```

把每块评论数改成 30。

```bash
uv run python scripts/run_pipeline.py --dry-run
```

只跑 ingest + chunk，不调用 LLM，不写入向量库。

### `scripts/streamlit_patterns.py`

用途：可视化查看 ChromaDB 中的 PatternCard，并测试检索效果。

启动：

```bash
uv run streamlit run scripts/streamlit_patterns.py
```

功能：
- 样本浏览
- 搜索与排序
- 检索测试
- 菜单预览

## 3. 配置方式

先复制环境变量模板：

```bash
cp .env.example .env
```

至少需要配置：

```env
LLM_API_KEY=你的提取模型 API key
EMBED_API_KEY=你的嵌入模型 API key
```

默认值：

- `LLM_API_BASE=https://api.openai.com/v1`
- `LLM_MODEL=gpt-4o-mini`
- `EMBED_API_BASE=https://api.siliconflow.cn/v1`
- `EMBED_MODEL=Qwen/Qwen3-Embedding-4B`
- `EMBED_DIMENSIONS=2048`

默认 Bilibili 数据库路径为：

```text
../raw-data-pipeline/bilibili/_storage/database/bilibili.db
```

如果数据库不在这个位置，可以在 `.env` 中覆盖：

```env
BILIBILI_DB_PATH=/your/path/to/bilibili.db
```

## 4. 推荐调用顺序

### 第一次验证

先确认测试通过：

```bash
uv run --with pytest pytest -q
```

再做 dry-run：

```bash
uv run python scripts/run_pipeline.py --dry-run --max-chunks 1
```

如果 dry-run 正常，再做真实提取：

```bash
uv run python scripts/run_pipeline.py --max-chunks 3
```

建议先从 `--max-chunks 1` 或 `--max-chunks 3` 开始，不要一上来放开所有评论块。

## 5. 本次已验证结果

已验证：

- `uv run --with pytest pytest -q` → **41 passed, 1 skipped**
- `uv run python scripts/run_pipeline.py --dry-run --max-chunks 1` → **可正常读取数据库、清洗评论、分块**

其中 `1 skipped` 是真实 LLM 集成测试，只有设置 `LLM_API_KEY` 后才会运行。

## 6. 目前还没做的部分

以下内容仍未实现：

- 对外 API 层（如 FastAPI）
- 和 `chat-core/bot` 的实际对接
- 真正在线对话时的工具调用闭环
- 提取 prompt 的效果调优

## 7. 下一步建议

建议按这个顺序继续：

1. 在 `.env` 里填好真实 API key
2. 跑 `uv run python scripts/run_pipeline.py --max-chunks 3`
3. 检查提取出的 PatternCard 质量
4. 根据结果微调 `src/brain/extract/refiner.py`
5. 再实现 API 层，把 compose 输出接到 bot
