# Streamlit 模式库查看器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 solution-brain 增加一个可直接运行的 Streamlit 界面，用于浏览已提取的 PatternCard、测试检索结果，并预览 compose menu。

**Architecture:** 采用单脚本入口 `scripts/streamlit_patterns.py`，复用现有 `PatternDB`、`retrieve_patterns()` 和 `build_menu()`。把与 UI 无关的筛选和排序逻辑提取为纯函数，便于测试，Streamlit 负责渲染与交互。

**Tech Stack:** Python 3.11, uv, Streamlit, ChromaDB, pytest

---

## File Structure

```text
solution-brain/
├── pyproject.toml                          # 新增 streamlit 依赖
├── scripts/
│   ├── run_pipeline.py
│   └── streamlit_patterns.py              # Streamlit 入口，样本浏览 + 检索测试 + 菜单预览
├── src/brain/
│   └── viewer.py                          # 纯函数：筛选、排序、摘要格式化
├── tests/
│   └── test_viewer.py                     # viewer 纯函数测试
└── docs/
    └── 2026-04-07-solution-brain-usage.md # 补充 Streamlit 启动方式
```

### 设计边界

- `scripts/streamlit_patterns.py` 只负责页面布局、输入控件、错误展示、调用已有模块
- `src/brain/viewer.py` 负责纯数据逻辑，避免把筛选/排序写死在 Streamlit 里无法测试
- 不改 `PatternDB` / `retriever` / `compose` 的现有接口
- 第一版不做编辑写入能力

---

## Task 1: 添加 viewer 纯函数与测试

**Files:**
- Create: `src/brain/viewer.py`
- Create: `tests/test_viewer.py`

- [ ] **Step 1: 写失败测试，覆盖搜索、排序和摘要格式化**

创建 `tests/test_viewer.py`：

```python
from datetime import datetime

from brain.models import FrequencyProfile, PatternCard
from brain.viewer import filter_patterns, format_pattern_summary, sort_patterns


def _make_card(
    id: str,
    title: str,
    *,
    description: str,
    template: str,
    examples: list[str],
    updated_day: int,
    recent: int,
    medium: int,
    total: int,
) -> PatternCard:
    return PatternCard(
        id=id,
        title=title,
        description=description,
        template=template,
        examples=examples,
        frequency=FrequencyProfile(
            recent=recent,
            medium=medium,
            long_term=total,
            total=total,
        ),
        source="test",
        created_at=datetime(2026, 4, 1),
        updated_at=datetime(2026, 4, updated_day),
    )


class TestFilterPatterns:
    def test_filter_patterns_matches_title(self):
        cards = [
            _make_card(
                "1",
                "好家伙式吐槽",
                description="对离谱情况吐槽",
                template="[A]...好家伙...",
                examples=["这也行...好家伙..."],
                updated_day=1,
                recent=4,
                medium=8,
                total=10,
            ),
            _make_card(
                "2",
                "反问句",
                description="用反问表达不满",
                template="这也叫[A]？",
                examples=["这也叫解释？"],
                updated_day=2,
                recent=2,
                medium=4,
                total=6,
            ),
        ]

        result = filter_patterns(cards, "好家伙")

        assert [card.id for card in result] == ["1"]

    def test_filter_patterns_matches_examples_and_description(self):
        cards = [
            _make_card(
                "1",
                "吐槽",
                description="对离谱情况吐槽",
                template="[A]...好家伙...",
                examples=["这也行...好家伙..."],
                updated_day=1,
                recent=4,
                medium=8,
                total=10,
            ),
            _make_card(
                "2",
                "撒娇",
                description="用叠字表达亲近",
                template="[A]啦~",
                examples=["拜托拜托啦~"],
                updated_day=2,
                recent=2,
                medium=4,
                total=6,
            ),
        ]

        by_description = filter_patterns(cards, "亲近")
        by_example = filter_patterns(cards, "拜托")

        assert [card.id for card in by_description] == ["2"]
        assert [card.id for card in by_example] == ["2"]

    def test_filter_patterns_blank_query_returns_all(self):
        cards = [
            _make_card(
                "1",
                "吐槽",
                description="对离谱情况吐槽",
                template="[A]...好家伙...",
                examples=["这也行...好家伙..."],
                updated_day=1,
                recent=4,
                medium=8,
                total=10,
            )
        ]

        result = filter_patterns(cards, "   ")

        assert [card.id for card in result] == ["1"]


class TestSortPatterns:
    def test_sort_patterns_by_updated_at_desc(self):
        cards = [
            _make_card(
                "1",
                "A",
                description="a",
                template="[A]",
                examples=["a"],
                updated_day=1,
                recent=1,
                medium=2,
                total=10,
            ),
            _make_card(
                "2",
                "B",
                description="b",
                template="[B]",
                examples=["b"],
                updated_day=7,
                recent=1,
                medium=2,
                total=10,
            ),
        ]

        result = sort_patterns(cards, "updated_at")

        assert [card.id for card in result] == ["2", "1"]

    def test_sort_patterns_by_freshness_desc(self):
        cards = [
            _make_card(
                "1",
                "A",
                description="a",
                template="[A]",
                examples=["a"],
                updated_day=1,
                recent=1,
                medium=2,
                total=10,
            ),
            _make_card(
                "2",
                "B",
                description="b",
                template="[B]",
                examples=["b"],
                updated_day=1,
                recent=8,
                medium=9,
                total=10,
            ),
        ]

        result = sort_patterns(cards, "freshness")

        assert [card.id for card in result] == ["2", "1"]

    def test_sort_patterns_by_title_asc(self):
        cards = [
            _make_card(
                "1",
                "吐槽",
                description="a",
                template="[A]",
                examples=["a"],
                updated_day=1,
                recent=1,
                medium=2,
                total=10,
            ),
            _make_card(
                "2",
                "反问",
                description="b",
                template="[B]",
                examples=["b"],
                updated_day=1,
                recent=8,
                medium=9,
                total=10,
            ),
        ]

        result = sort_patterns(cards, "title")

        assert [card.title for card in result] == ["反问", "吐槽"]


class TestFormatPatternSummary:
    def test_format_pattern_summary_contains_key_fields(self):
        card = _make_card(
            "1",
            "好家伙式吐槽",
            description="对离谱情况吐槽",
            template="[A]...好家伙...",
            examples=["这也行...好家伙..."],
            updated_day=3,
            recent=4,
            medium=8,
            total=10,
        )

        summary = format_pattern_summary(card)

        assert "好家伙式吐槽" in summary
        assert "[A]...好家伙..." in summary
        assert "freshness=" in summary
        assert "total=10" in summary
```

- [ ] **Step 2: 运行测试，确认失败**

运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run --with pytest pytest tests/test_viewer.py -q
```

预期：失败，提示 `ModuleNotFoundError: No module named 'brain.viewer'`。

- [ ] **Step 3: 写最小实现**

创建 `src/brain/viewer.py`：

```python
from __future__ import annotations

from brain.models import PatternCard


_SORT_OPTIONS = {"updated_at", "freshness", "title"}


def filter_patterns(patterns: list[PatternCard], query: str) -> list[PatternCard]:
    needle = query.strip().lower()
    if not needle:
        return list(patterns)

    result = []
    for card in patterns:
        haystack = "\n".join(
            [
                card.title,
                card.template,
                card.description,
                *card.examples,
            ]
        ).lower()
        if needle in haystack:
            result.append(card)
    return result


def sort_patterns(patterns: list[PatternCard], sort_by: str) -> list[PatternCard]:
    if sort_by not in _SORT_OPTIONS:
        sort_by = "updated_at"

    if sort_by == "title":
        return sorted(patterns, key=lambda card: card.title)
    if sort_by == "freshness":
        return sorted(
            patterns,
            key=lambda card: card.frequency.freshness,
            reverse=True,
        )
    return sorted(patterns, key=lambda card: card.updated_at, reverse=True)


def format_pattern_summary(card: PatternCard) -> str:
    return (
        f"{card.title} | {card.template} | "
        f"freshness={card.frequency.freshness:.2f} | total={card.frequency.total}"
    )
```

- [ ] **Step 4: 运行测试，确认通过**

运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run --with pytest pytest tests/test_viewer.py -q
```

预期：`7 passed`。

- [ ] **Step 5: 提交**

```bash
git -C /home/sparidae/projects/silicachan/solution-brain add src/brain/viewer.py tests/test_viewer.py
git -C /home/sparidae/projects/silicachan/solution-brain commit -m "$(cat <<'EOF'
feat(brain): 添加模式查看器的数据筛选与排序逻辑

新增 viewer 纯函数，支持 PatternCard 的搜索、排序和摘要格式化。
为后续 Streamlit 页面提供可测试的数据处理基础。
EOF
)"
```

---

## Task 2: 添加 Streamlit 入口并实现样本浏览

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `scripts/streamlit_patterns.py`

- [ ] **Step 1: 写失败测试，先验证依赖尚未安装**

运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run python -c "import streamlit"
```

预期：失败，提示 `ModuleNotFoundError: No module named 'streamlit'`。

- [ ] **Step 2: 添加依赖**

修改 `pyproject.toml` 中的依赖列表：

```toml
[project]
dependencies = [
    "chromadb>=1.0.0",
    "openai>=1.0.0",
    "jinja2>=3.1.0",
    "python-dotenv>=1.0.0",
    "streamlit>=1.44.0",
]
```

然后运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv sync
```

预期：`uv.lock` 更新完成。

- [ ] **Step 3: 创建 Streamlit 页面骨架并实现样本浏览**

创建 `scripts/streamlit_patterns.py`：

```python
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain.config import CHROMA_DIR, RETRIEVAL_TOP_K
from brain.store.pattern_db import PatternDB
from brain.viewer import filter_patterns, format_pattern_summary, sort_patterns


st.set_page_config(page_title="solution-brain 模式库", layout="wide")
st.title("solution-brain 模式库查看器")
st.caption("查看 PatternCard 样本，检查提取质量")


@st.cache_resource
def get_db() -> PatternDB:
    return PatternDB(CHROMA_DIR)


def render_pattern_card(card):
    with st.expander(format_pattern_summary(card), expanded=False):
        st.markdown(f"**ID**: `{card.id}`")
        st.markdown(f"**标题**: {card.title}")
        st.markdown(f"**模板**: `{card.template}`")
        st.markdown(f"**描述**: {card.description}")
        st.markdown(f"**来源**: {card.source}")
        st.markdown(
            "**频率**: "
            f"recent={card.frequency.recent}, "
            f"medium={card.frequency.medium}, "
            f"long_term={card.frequency.long_term}, "
            f"total={card.frequency.total}, "
            f"freshness={card.frequency.freshness:.2f}"
        )
        st.markdown(
            f"**时间**: created_at={card.created_at.isoformat()} | "
            f"updated_at={card.updated_at.isoformat()}"
        )
        st.markdown("**例句**")
        for example in card.examples:
            st.markdown(f"- {example}")


def main() -> None:
    try:
        db = get_db()
        all_patterns = db.list_all()
    except Exception as exc:
        st.error(f"读取 ChromaDB 失败：{exc}")
        return

    st.metric("PatternCard 总数", len(all_patterns))

    if not all_patterns:
        st.info("当前还没有 PatternCard。请先运行 scripts/run_pipeline.py 生成数据。")
        return

    search_query = st.text_input("搜索 PatternCard", placeholder="按标题、模板、描述或例句搜索")
    sort_by = st.selectbox(
        "排序方式",
        options=["updated_at", "freshness", "title"],
        format_func=lambda x: {
            "updated_at": "最近更新",
            "freshness": "热度 freshness",
            "title": "标题",
        }[x],
    )

    visible_patterns = sort_patterns(filter_patterns(all_patterns, search_query), sort_by)
    st.write(f"当前显示 {len(visible_patterns)} 条")

    for card in visible_patterns:
        render_pattern_card(card)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 启动页面，确认可打开并显示样本**

运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run streamlit run scripts/streamlit_patterns.py
```

预期：浏览器中能打开页面，看到 PatternCard 总数、搜索框、排序框和样本详情列表。

- [ ] **Step 5: 提交**

```bash
git -C /home/sparidae/projects/silicachan/solution-brain add pyproject.toml uv.lock scripts/streamlit_patterns.py
git -C /home/sparidae/projects/silicachan/solution-brain commit -m "$(cat <<'EOF'
feat(brain): 添加 Streamlit 模式样本浏览界面

新增 streamlit 入口脚本，支持浏览 ChromaDB 中的 PatternCard。
首版提供数量统计、搜索、排序和详情展开查看能力。
EOF
)"
```

---

## Task 3: 为页面增加检索测试与菜单预览

**Files:**
- Modify: `scripts/streamlit_patterns.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: 为检索基础行为补测试**

在 `tests/test_store.py` 追加：

```python
    def test_retrieve_respects_top_k(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        db.save([
            _make_card("1", "吐槽表达", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "撒娇语气", "用叠字和语气词表达撒娇"),
            _make_card("3", "反问句式", "用反问表达不满"),
        ])

        results = retrieve_patterns(db, "这也太离谱了吧", top_k=1)

        assert len(results) == 1
```

- [ ] **Step 2: 运行测试，确认通过**

运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run --with pytest pytest tests/test_store.py -q
```

预期：全部通过。

- [ ] **Step 3: 在 Streamlit 页面中加入检索测试区**

修改 `scripts/streamlit_patterns.py`，将主页面拆成两个 tab：

```python
from brain.compose.menu import build_menu
from brain.store.retriever import retrieve_patterns
```

并在 `main()` 中加入：

```python
    browse_tab, retrieve_tab = st.tabs(["样本浏览", "检索测试"])

    with browse_tab:
        search_query = st.text_input("搜索 PatternCard", placeholder="按标题、模板、描述或例句搜索")
        sort_by = st.selectbox(
            "排序方式",
            options=["updated_at", "freshness", "title"],
            format_func=lambda x: {
                "updated_at": "最近更新",
                "freshness": "热度 freshness",
                "title": "标题",
            }[x],
        )

        visible_patterns = sort_patterns(filter_patterns(all_patterns, search_query), sort_by)
        st.write(f"当前显示 {len(visible_patterns)} 条")

        for card in visible_patterns:
            render_pattern_card(card)

    with retrieve_tab:
        query_text = st.text_area(
            "输入检索文本",
            placeholder="例如：这也太离谱了吧，我直接好家伙",
            height=120,
        )
        top_k = st.number_input("返回条数 top_k", min_value=1, max_value=20, value=RETRIEVAL_TOP_K)

        if st.button("执行检索"):
            if not query_text.strip():
                st.warning("请输入检索文本。")
            else:
                try:
                    results = retrieve_patterns(db, query_text, top_k=int(top_k))
                except Exception as exc:
                    st.error(f"检索失败：{exc}")
                else:
                    if not results:
                        st.info("没有匹配到模式。")
                    else:
                        st.subheader("检索结果")
                        for card in results:
                            render_pattern_card(card)

                        st.subheader("菜单预览")
                        st.code(build_menu(results), language="text")
```

- [ ] **Step 4: 手动验证检索区**

运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run streamlit run scripts/streamlit_patterns.py
```

在页面中输入：

```text
这也太离谱了吧，我直接好家伙
```

预期：
- 检索结果区展示至少 1 条 PatternCard（如果当前库里已有相关样本）
- 菜单预览区显示 `build_menu()` 生成的文本
- 如果当前数据不足，也应稳定显示“没有匹配到模式”而不是报错

- [ ] **Step 5: 提交**

```bash
git -C /home/sparidae/projects/silicachan/solution-brain add scripts/streamlit_patterns.py tests/test_store.py
git -C /home/sparidae/projects/silicachan/solution-brain commit -m "$(cat <<'EOF'
feat(brain): 为模式查看器添加检索测试与菜单预览

支持在界面中输入查询文本，查看 retriever 返回的 PatternCard。
同时展示 compose menu 预览，便于检查最终注入效果。
EOF
)"
```

---

## Task 4: 更新使用文档并做最终验证

**Files:**
- Modify: `docs/2026-04-07-solution-brain-usage.md`

- [ ] **Step 1: 更新使用说明文档**

在 `docs/2026-04-07-solution-brain-usage.md` 的“当前可运行的脚本”部分补充：

```markdown
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
```

- [ ] **Step 2: 跑完整测试集**

运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run --with pytest pytest -q
```

预期：全部通过。

- [ ] **Step 3: 最终手动验证 UI 启动**

运行：

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run streamlit run scripts/streamlit_patterns.py
```

预期：
- 页面可正常打开
- 样本浏览可用
- 检索测试可用
- 菜单预览可用

- [ ] **Step 4: 提交**

```bash
git -C /home/sparidae/projects/silicachan/solution-brain add docs/2026-04-07-solution-brain-usage.md
git -C /home/sparidae/projects/silicachan/solution-brain commit -m "$(cat <<'EOF'
doc(brain): 补充 Streamlit 模式查看器使用说明

在使用文档中加入 Streamlit 查看器的启动方式和功能说明。
便于后续直接浏览样本和测试检索效果。
EOF
)"
```

---

## Self-Review

### Spec coverage
- 样本浏览：Task 2 实现
- 搜索 / 排序：Task 1 + Task 2 实现
- 检索测试：Task 3 实现
- 菜单预览：Task 3 实现
- 启动方式和仓库自洽：Task 2 与 Task 4 实现
- 文档更新：Task 4 实现

### Placeholder scan
- 未使用 TBD / TODO / “自行实现” 类占位语
- 每个代码步骤都给了明确代码或命令
- 每个任务都包含验证与提交步骤

### Type consistency
- `filter_patterns` / `sort_patterns` / `format_pattern_summary` 只在 `src/brain/viewer.py` 定义，后续步骤保持一致
- `retrieve_patterns`、`build_menu`、`PatternDB` 均复用现有接口，未引入新命名漂移
