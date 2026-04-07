# solution-brain Streamlit 模式库查看器设计

> 日期：2026-04-07
> 目标：为 solution-brain 增加一个自洽、易调用的 Streamlit 界面，用于查看已提取的 PatternCard 样本，并测试检索效果。

## 1. 背景与目标

当前 `solution-brain` 已完成 MVP pipeline：

- `ingest`：读取 Bilibili 评论
- `extract`：调用 LLM 提取语言模式
- `store`：将 PatternCard 存入 ChromaDB
- `compose`：组装菜单与工具定义
- `scripts/run_pipeline.py`：运行全流程

目前缺少一个方便人工检查结果的可视化入口。用户当前最关心的是两件事：

1. 已经写入库里的 PatternCard 长什么样，质量如何
2. 给一段查询文本时，retriever 会返回哪些模式

因此第一版界面不做编辑、不做管理后台，而是聚焦于**模式样本浏览 + 检索测试**。

## 2. 设计原则

- **与仓库结构自洽**：延续 `scripts/` 作为可直接运行入口的习惯
- **复用现有业务代码**：只调用 `src/brain/store`、`src/brain/compose` 等现有模块，不复制逻辑
- **先做查看，不做修改**：第一版是调试/验收工具，不承担写操作
- **调用简单**：用户应能直接运行 `uv run streamlit run ...`
- **最小但完整**：既能看样本，也能顺手测检索，不做额外抽象

## 3. 方案选择

### 方案 A：`scripts/streamlit_patterns.py` 单文件入口（采用）

在 `scripts/` 下增加一个 Streamlit 入口脚本，页面逻辑集中在这一处，内部直接调用现有业务模块。

**优点：**
- 跟当前 `scripts/run_pipeline.py` 的使用方式一致
- 最少新增结构，方便直接调用
- 适合作为当前 MVP 阶段的调试工具

**缺点：**
- 页面逻辑会集中在一个文件中，后续功能扩大时可能需要拆分

### 方案 B：`src/brain/ui/` + `scripts/run_ui.py`

更规整，但对当前范围偏重，不符合第一版“先能看”的目标。

### 方案 C：独立 viewer 子项目

隔离最强，但明显超出当前需求，属于过度设计。

**结论：** 第一版采用 **方案 A**。

## 4. 文件与位置

新增文件：

```text
solution-brain/
├── scripts/
│   ├── run_pipeline.py
│   └── streamlit_patterns.py   # 新增：PatternCard 可视化查看器
```

依赖变更：
- 在 `pyproject.toml` 中加入 `streamlit`

文档更新：
- 在使用说明中补充启动方法

## 5. 启动方式

```bash
uv run streamlit run scripts/streamlit_patterns.py
```

这与当前仓库里 `scripts/run_pipeline.py` 的调用方式一致，符合“脚本入口在 `scripts/` 下”的现有逻辑。

## 6. 页面结构

第一版页面分为两个主要区域。

### 6.1 模式样本浏览区

用途：查看已写入 ChromaDB 的 PatternCard 样本。

功能：
- 显示当前 PatternCard 总数
- 文本搜索：按 `title`、`template`、`description`、`examples` 做包含过滤
- 排序方式：
  - `updated_at`（最近更新优先）
  - `freshness`（热度优先）
  - `title`（字典序）
- 列表展示卡片摘要：
  - title
  - template
  - freshness
  - total frequency
- 展开后显示详情：
  - id
  - description
  - examples
  - source
  - recent / medium / long_term / total
  - created_at / updated_at

### 6.2 检索测试区

用途：输入一段查询文本，观察 retriever 会返回什么 PatternCard。

功能：
- 文本输入框：输入用户消息、对话上下文、测试 query
- `top_k` 可调（默认沿用配置）
- 调用现有 `retrieve_patterns(...)`
- 展示检索结果卡片：
  - title
  - template
  - freshness
  - description
  - examples
- 额外展示 compose menu 预览：
  - 直接调用现有 `build_menu(...)`
  - 用于观察这些结果在最终 prompt 中会以什么样子出现

## 7. 数据流设计

### 样本浏览

```text
Streamlit UI
  → PatternDB.list_all()
  → 本地过滤 / 排序
  → 渲染列表与详情
```

### 检索测试

```text
Streamlit UI 输入 query
  → retrieve_patterns(db, query, top_k)
  → 返回 PatternCard 列表
  → build_menu(patterns)
  → 渲染检索结果 + 菜单预览
```

### 关键约束

- UI 不直接操作底层 Chroma collection
- 所有读取都经由现有 `PatternDB` / `retriever` / `compose` 模块
- 不新增新的存储格式
- 不修改现有 pipeline 产物

## 8. 边界与非目标

第一版**明确不做**：

- 编辑 PatternCard
- 删除 PatternCard
- 合并 / 修正 PatternCard
- 在 UI 中触发 pipeline
- 多页面路由
- 用户鉴权
- 独立后端服务

这样可以保证这个界面保持“只读质检工具”的角色，而不是演变成管理后台。

## 9. 错误处理

界面需要处理以下场景：

1. **数据库为空**
   - 提示当前没有 PatternCard，建议先运行 `scripts/run_pipeline.py`

2. **ChromaDB 目录不存在或读取失败**
   - 显示明确报错，不隐藏异常原因

3. **检索 query 为空**
   - 不执行检索，只提示用户输入文本

4. **检索无结果**
   - 正常显示“无匹配模式”，而不是报错

## 10. 测试与验证

实现后验证分三层：

1. **基础运行验证**
   - `uv run streamlit run scripts/streamlit_patterns.py`
   - 页面可正常打开

2. **数据浏览验证**
   - 能展示当前已提取的 4 个 PatternCard
   - 搜索与排序生效

3. **检索验证**
   - 输入一段测试文本，可返回 PatternCard
   - 菜单预览可正确显示

如果需要补自动化测试，优先把可复用的筛选/排序逻辑抽成普通函数后再测；第一版不强求为 Streamlit 视图层写 UI 自动化。

## 11. 后续扩展路径

如果这套查看器后续证明常用，再考虑第二阶段扩展：

- 按 source / 时间范围过滤
- 查看单卡片原始 JSON
- 导出 JSON / CSV
- 添加“相似 pattern 对比”视图
- 增加 prompt 预览完整页

这些都可以在当前入口上渐进添加，不需要推翻第一版结构。

## 12. 最终结论

第一版实现一个只读 Streamlit 查看器，文件放在：

- `scripts/streamlit_patterns.py`

调用方式：

```bash
uv run streamlit run scripts/streamlit_patterns.py
```

范围限定为：
- **样本浏览**
- **检索测试**
- **菜单预览**

不做编辑与管理能力。

这个方案与 `solution-brain` 当前仓库结构、脚本入口风格、MVP 阶段目标保持一致，足够简单，也方便你后续直接调用。