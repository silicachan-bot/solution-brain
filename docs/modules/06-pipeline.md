# pipeline 模块

这里的 `pipeline` 不是 `src/brain/` 下的一个 package，而是当前项目把多个模块串起来执行的命令行入口，主要对应 [scripts/run_pipeline.py](../../scripts/run_pipeline.py)。

## 1. 模块职责

`pipeline` 当前负责：

1. 解析命令行参数
2. 初始化 ingest / extract / store 所需对象
3. 执行整条离线提取链路
4. 在成功完成后写回水位线
5. 在终端打印进度

它不负责：
- 提供 API
- 提供调度服务
- 并发执行

## 2. 代码位置

- 主入口：[scripts/run_pipeline.py](../../scripts/run_pipeline.py)

关键代码区段：
- 参数解析：[scripts/run_pipeline.py:27-33](../../scripts/run_pipeline.py#L27-L33)
- 初始化依赖：[scripts/run_pipeline.py:35-37](../../scripts/run_pipeline.py#L35-L37)
- 视频过滤：[scripts/run_pipeline.py:39-55](../../scripts/run_pipeline.py#L39-L55)
- 逐视频处理：[scripts/run_pipeline.py:57-84](../../scripts/run_pipeline.py#L57-L84)
- 去重入库：[scripts/run_pipeline.py:90-99](../../scripts/run_pipeline.py#L90-L99)
- 更新水位线：[scripts/run_pipeline.py:101-105](../../scripts/run_pipeline.py#L101-L105)

## 3. 核心对象 / 函数

### 3.1 `main()`

整个脚本只有一个主函数 `main()`，定义在 [scripts/run_pipeline.py:27-108](../../scripts/run_pipeline.py#L27-L108)。

它串联的对象包括：
- `BilibiliReader`
- `WatermarkState`
- `PatternDB`
- `QwenEmbedder`

以及函数：
- `clean_comments()`
- `chunk_comments()`
- `extract_from_chunk()`
- `deduplicate_and_merge()`

### 3.2 命令行参数

当前支持：

- `--full`
  - 忽略水位线，全量重跑
- `--chunk-size`
  - 覆盖默认分块大小
- `--max-chunks`
  - 限制每个视频最多处理多少块评论对
- `--dry-run`
  - 只跑到分块阶段，不调用 LLM，不写库

参数定义见 [scripts/run_pipeline.py:31-36](../../scripts/run_pipeline.py#L31-L36)。

## 4. 输入与输出

### 4.1 输入

- 命令行参数
- Bilibili SQLite 数据库
- `.env` 中的 API / 路径配置
- 本地 `state.json`
- 本地 LanceDB 目录（`LANCEDB_DIR`）

### 4.2 输出

- 新增或更新后的 LanceDB 表
- 更新后的 `state.json`
- Rich Live 实时终端界面（双进度条 + 统计行 + 已完成视频表格）
- `data/llm_responses.log`：每次 LLM 调用的完整 prompt 与回复

在 `--dry-run` 模式下：
- 不调用 LLM
- 不写入 LanceDB
- 不更新水位线
- 使用纯文本输出，不启动 Rich Live

## 5. 执行流程

当前脚本的实际执行顺序如下：

```text
1. 解析 CLI 参数
2. 初始化 reader / state / embedder / db（PatternDB(LANCEDB_DIR, embedder=embedder)）
3. 读取视频列表
4. 如果不是 --full，则按 watermark 过滤视频
5. 对每个视频：
   5.1 读取评论
   5.2 清洗评论
   5.3 构造评论对并分块
   5.4 如果设置了 --max-chunks，则截断当前视频的块数
   5.5 若不是 dry-run，则逐块调 LLM 提取
   5.6 对当前视频提取结果执行去重并立即写库
   5.7 更新当前视频的 watermark，支持断点续跑
6. 打印完成信息与 db.count() 最终总数
```

其中 dry-run 分支在 [scripts/run_pipeline.py:58-73](../../scripts/run_pipeline.py#L58-L73)。

## 6. 依赖关系

直接依赖：
- `brain.config`（含 `LANCEDB_DIR`）
- `brain.ingest.reader`
- `brain.ingest.cleaner`
- `brain.ingest.state`
- `brain.extract.chunker`
- `brain.extract.refiner`
- `brain.store.pattern_db`
- `brain.store.embedding`（`QwenEmbedder`）

对运行环境的隐含依赖：
- 能访问 Bilibili SQLite 文件
- LLM API 可用
- embedding API 可用
- 本地可写 `data/` 目录

## 7. 当前限制 / 已知偏差

1. 当前 pipeline 是单进程、串行的。
   - 每个视频、每个 chunk 都按顺序处理。

2. 错误处理比较直接。
   - 某次 LLM 或向量库调用抛异常时，脚本没有做复杂恢复逻辑。

3. 水位线只在整批完成后写一次。
   - 中途失败时，不会保留部分进度。

4. 终端使用 Rich Live 实时展示，不是结构化日志。
   - 适合手工运行观察；LLM 完整回复另存于 `data/llm_responses.log`。

5. `--dry-run` 统计”有评论的视频数”时，会重新读评论：
   - 这是为了简单实现，不是最省 IO 的写法。
