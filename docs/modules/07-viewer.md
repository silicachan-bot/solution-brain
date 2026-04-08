# viewer 模块

这里的 `viewer` 指当前项目中用于人工浏览 PatternCard 和手工测试检索效果的查看器实现，主要由两部分组成：

- [scripts/streamlit_patterns.py](../../scripts/streamlit_patterns.py)
- [src/brain/viewer.py](../../src/brain/viewer.py)

## 1. 模块职责

`viewer` 当前负责：

1. 把模式库内容可视化展示出来
2. 提供本地搜索和排序
3. 提供手动检索测试入口
4. 预览 `build_menu()` 生成的菜单文本

它不负责：
- 修改 PatternCard
- 回写数据库
- 在线聊天调用

## 2. 代码位置

- Streamlit 应用：[scripts/streamlit_patterns.py](../../scripts/streamlit_patterns.py)
- 浏览辅助函数：[src/brain/viewer.py](../../src/brain/viewer.py)

关键函数：
- `get_db()`：[scripts/streamlit_patterns.py:21-24](../../scripts/streamlit_patterns.py#L21-L24)
- `render_pattern_card()`：[scripts/streamlit_patterns.py:26-48](../../scripts/streamlit_patterns.py#L26-L48)
- `main()`：[scripts/streamlit_patterns.py:50-124](../../scripts/streamlit_patterns.py#L50-L124)
- `filter_patterns()`：[src/brain/viewer.py:8-25](../../src/brain/viewer.py#L8-L25)
- `sort_patterns()`：[src/brain/viewer.py:29-41](../../src/brain/viewer.py#L29-L41)
- `format_pattern_summary()`：[src/brain/viewer.py:45-49](../../src/brain/viewer.py#L45-L49)

## 3. 核心对象 / 函数

### 3.1 `get_db()`

使用 `@st.cache_resource` 包装，返回 `PatternDB(CHROMA_DIR)`。

这意味着：
- 查看器会复用同一个数据库连接对象
- 默认读取本地 `data/chromadb`

### 3.2 `render_pattern_card(card)`

负责把单个 `PatternCard` 展开成可读界面，展示内容包括：
- ID
- 标题
- 模板
- 描述
- 来源
- 频率统计与 freshness
- 创建/更新时间
- 例句列表

### 3.3 `filter_patterns(patterns, query)`

定义在 [src/brain/viewer.py:8-25](../../src/brain/viewer.py#L8-L25)。

行为：
- 把 `title`、`template`、`description`、`examples` 拼成一个搜索串
- 全部转小写后做子串匹配
- 返回过滤后的列表

这是纯本地内存过滤，不走向量检索。

### 3.4 `sort_patterns(patterns, sort_by)`

定义在 [src/brain/viewer.py:29-41](../../src/brain/viewer.py#L29-L41)。

支持三种排序键：
- `updated_at`
- `freshness`
- `title`

非法排序键会回退到 `updated_at`。

### 3.5 `format_pattern_summary(card)`

定义在 [src/brain/viewer.py:45-49](../../src/brain/viewer.py#L45-L49)。

生成 expander 标题，格式类似：

```text
标题 | 模板 | freshness=0.xx | total=xx
```

## 4. 输入与输出

### 4.1 浏览页

输入：
- `db.list_all()` 返回的全部 `PatternCard`
- 用户输入的搜索词与排序方式

输出：
- 过滤、排序后的卡片列表 UI

### 4.2 检索测试页

输入：
- 用户输入的 query text
- `top_k`

输出：
- `retrieve_patterns()` 返回的结果卡片 UI
- `build_menu(results)` 生成的菜单文本

## 5. 执行流程

### 5.1 启动流程

```text
streamlit run scripts/streamlit_patterns.py
  -> 初始化 PatternDB
  -> list_all()
  -> 渲染两个 tab
```

### 5.2 样本浏览流程

```text
全部 PatternCard
  -> filter_patterns(...)
  -> sort_patterns(...)
  -> render_pattern_card(...) 逐条展示
```

### 5.3 检索测试流程

```text
输入 query_text
  -> retrieve_patterns(db, query_text, top_k)
  -> render_pattern_card(...) 展示结果
  -> build_menu(results) 预览菜单
```

## 6. 依赖关系

依赖：
- `streamlit`
- [src/brain/store/pattern_db.py](../../src/brain/store/pattern_db.py)
- [src/brain/store/retriever.py](../../src/brain/store/retriever.py)
- [src/brain/compose/menu.py](../../src/brain/compose/menu.py)
- [src/brain/config.py](../../src/brain/config.py)
- [src/brain/viewer.py](../../src/brain/viewer.py)

## 7. 当前限制 / 已知偏差

1. 查看器当前是只读的。
   - 没有编辑、删除、标注质量等管理能力。

2. 浏览页使用 `list_all()` 全量载入。
   - 当前数据量下可接受，但规模大了会变重。

3. 本地搜索是简单子串匹配。
   - 不支持分词、模糊匹配或高亮。

4. 检索测试页调用的是当前真实检索逻辑。
   - 所以结果同时受向量库内容、embedding 配置和 `freshness` 权重影响。

5. UI 主要服务开发调试。
   - 现在没有用户权限、审计或多用户协作概念。
