# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See @README.md for project overview and available scripts.

## 模块文档

- 整体架构与数据流: @docs/modules/00-overview.md
- 公共数据结构: @docs/modules/01-shared.md
- 数据摄取层: @docs/modules/02-ingest.md
- 模式提取层: @docs/modules/03-extract.md
- 向量存储层: @docs/modules/04-store.md
- Prompt 组装层: @docs/modules/05-compose.md
- Pipeline 入口: @docs/modules/06-pipeline.md
- Streamlit 查看器: @docs/modules/07-viewer.md

## 测试

```bash
uv run pytest tests/
uv run pytest tests/test_ingest.py  # 单文件
```
