# Pattern Extract Loop

这个目录用于做”模式提取模块”的循环迭代测试。

## 迭代节奏

**第一阶段（Round 03 起）：期望驱动**
1. 从数据库里选 3 个评论对较少的视频，列出全部评论对
2. 用户在样本文档里手写期望模式（自然语言，不需要 JSON）
3. 清空数据库，只对这 3 个视频跑提取
4. 输出对比文档：期望 vs 实际，逐条列出
5. 根据差距迭代 prompt/代码，每次改动提交一次
6. 循环直到基本匹配

**第二阶段：扩展到 10 个视频**
- 一轮匹配之后，扩展到 10 个视频重跑
- 提供下一轮反馈文档

**第三阶段：再扩展 10 个视频**

## 快捷命令

```bash
# 清空 data/ 并对指定视频跑提取
bash scripts/clear_and_eval.sh BV11qx4zAEPz BV128GbzBEjD BV11MbhzJEtb

# 全量 pipeline（不清库）
uv run python scripts/run_pipeline.py --limit 3 --full
```

## 文档列表

### Round 03（当前）

- [round-03-samples.md](./round-03-samples.md) — 3 个视频评论对 + **用户待填期望模式**

### Round 01-02（旧记录）

- [round-01-BV1f7wtzaExX.md](./round-01-BV1f7wtzaExX.md)
- [round-01-BV1c1wWzoELy.md](./round-01-BV1c1wWzoELy.md)
- [round-01-BV1HUckzCEwg.md](./round-01-BV1HUckzCEwg.md)
- [round-02-feedback.md](./round-02-feedback.md)
