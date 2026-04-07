# Brain

智能体层：RAG、记忆系统、语言模式数据库、Prompt 工程。

为 `bot` 提供基于强基模的智能服务（区别于微调路线的直接推理）。

## 当前可用能力

- 从 Bilibili 评论中提取语言模式
- 将 PatternCard 存入 ChromaDB
- 基于相似度与 freshness 检索模式
- 组装 compose menu 供后续注入使用
- 提供 Streamlit 可视化查看器浏览样本与测试检索

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
