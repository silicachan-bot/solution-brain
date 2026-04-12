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

## 迭代工作流

- 在迭代 worktree 中，每完成一轮“可运行、已验证”的完整修改后，必须立即提交一次 git commit。
- worktree 分支上的 commit message 一律使用中文，且需要详细说明本轮改动点、验证方式、以及与上一轮相比重点调整了什么。
- 不要把一次完整版本长时间停留在未提交状态；默认每轮迭代都应有可回退的提交记录。
- 新建迭代 worktree 时，统一创建在原项目的 `.claude/worktrees/` 目录下，不再散落在仓库外部同级目录。
- 做 prompt 迭代测试时，只看当前 worktree 自己生成的数据，不引用 main 工作区已有数据库作为评估依据。
- 每次重新验证提取效果前，先清空当前 worktree 的 `data/` 运行产物，再从空库重新跑，避免旧数据污染本轮判断。
