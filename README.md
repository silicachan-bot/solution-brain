# Brain

智能体层：RAG、记忆系统、语言模式数据库、Prompt 工程。

为 `bot` 提供基于强基模的智能服务（区别于微调路线的直接推理）。

## 当前可用能力

- 从 Bilibili 评论中提取语言模式
- 将 PatternCard 存入 LanceDB（双路向量）
- 基于相似度与 freshness 检索模式
- 组装 compose menu 供后续注入使用
- 提供 Streamlit 可视化查看器浏览样本与测试检索

## 核心设计

### 双路向量存储

每张 `PatternCard` 写入时同时生成两条独立向量：

| 向量列 | embedding 来源 | 用途 |
|---|---|---|
| `vec_template` | `template`（模板句式） | 去重阶段：检索结构相近的同类模式 |
| `vec_semantic` | `description` + `examples` 拼接 | 检索阶段：按语义内容找匹配模式 |

两者面向不同场景：**去重关心句式结构是否已存在，检索关心语境语义是否匹配**。

### 两阶段去重

提取完成后，新模式经过两轮去重再入库：

1. **批次内去重**：对当次提取的模式两两做双路向量检索，合并候选后交 LLM 判断是否重复
2. **入库前去重**：与持久库做双路向量检索，重复的合并频率数据后更新已有记录，不重复的新增

两轮均先按可配置相似度阈值过滤候选，再使用"Top-N 候选 + LLM 判断"方案。默认阈值为 `0.8`，低于阈值的候选直接视为不重复。

## 快速开始

安装依赖：

```bash
uv sync
```

复制环境变量模板：

```bash
cp .env.example .env
```

至少配置：

```env
LLM_API_KEY=你的提取模型 API key
EMBED_API_KEY=你的嵌入模型 API key
```

## 可运行脚本

### 1. 运行提取 pipeline

```bash
uv run python scripts/run_pipeline.py --limit 3
```

### 2. 打开 PatternCard 查看器

```bash
uv run streamlit run scripts/streamlit_patterns.py
```

可用于：
- 浏览样本
- 搜索与排序
- 检索测试
- 菜单预览

## 规划功能

- 语言模式采集与总结
- RAG 检索增强生成
- 对话记忆管理
- Prompt 注入与编排

## 相关文档

- `docs/2026-04-07-solution-brain-usage.md`
- `docs/2026-04-07-streamlit-pattern-viewer-design.md`
- `docs/plans/2026-04-07-streamlit-pattern-viewer.md`
