# 相似度去重 + LanceDB 替换 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LanceDB 替换 ChromaDB，利用多向量列原生支持实现双路检索（`vec_template` + `vec_semantic`）；用 Top-N 检索 + LLM 判断替换标题精确匹配去重；同步删除 `title` 字段。

**Architecture:** LanceDB 表存储两个独立向量列——`vec_template`（template 的 embedding，去重用）和 `vec_semantic`（description + examples 的 embedding，语境检索用）。去重时分别查两列取 top-N，合并候选后交 LLM 判断，日志并列输出两路结果。`PatternDB` 重写为 LanceDB 封装，对外接口基本不变（save/update/get/list_all/query_by_template/query_by_semantic）。`QwenEmbeddingFunction` 改为 `QwenEmbedder`，提供 `embed(texts) -> list[list[float]]`。

**Tech Stack:** lancedb, pyarrow, openai-compatible SDK, rich.console, pytest

---

## 文件变更一览

| 动作 | 文件 | 说明 |
|------|------|------|
| 修改 | `pyproject.toml` | chromadb → lancedb |
| 修改 | `src/brain/models.py` | 删除 `title`，新增 `embed_text()` |
| 修改 | `src/brain/config.py` | `CHROMA_DIR` → `LANCEDB_DIR`，新增 `DEDUP_TOP_N` |
| 重写 | `src/brain/store/embedding.py` | `QwenEmbeddingFunction` → `QwenEmbedder` |
| 重写 | `src/brain/store/pattern_db.py` | ChromaDB → LanceDB，双向量列 |
| 修改 | `src/brain/store/retriever.py` | 改用 `query_by_semantic` |
| 修改 | `src/brain/extract/refiner.py` | 去掉 title、新增三个去重函数、替换旧 merge |
| 修改 | `src/brain/compose/menu.py` | 展示改用 template |
| 修改 | `src/brain/compose/tools.py` | 去掉 title 引用 |
| 修改 | `src/brain/viewer.py` | 过滤/排序/摘要改用 template |
| 修改 | `scripts/streamlit_patterns.py` | 去掉标题展示、排序改 template、config 引用更新 |
| 修改 | `scripts/run_pipeline.py` | 更新 import 和调用 |
| 重写 | `tests/test_store.py` | 适配 LanceDB + 删 title |
| 修改 | `tests/test_extract.py` | 修复损坏测试 + 删 title + 新增去重测试 |
| 修改 | `tests/test_compose.py` | 适配 PatternDB 新构造 + 删 title |

---

## Task 1: 基础改造（依赖、模型、配置）

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/brain/models.py`
- Modify: `src/brain/config.py`

- [ ] **Step 1: 替换 pyproject.toml 中的依赖**

把 `pyproject.toml` 中 `"chromadb>=1.0.0"` 替换为 `"lancedb>=0.20.0"`：

```toml
dependencies = [
    "lancedb>=0.20.0",
    "openai>=1.0.0",
    "jinja2>=3.1.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
    "streamlit>=1.44.0",
]
```

- [ ] **Step 2: 安装新依赖**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv sync
```

期望: 安装成功，lancedb 可用。

- [ ] **Step 3: 从 `PatternCard` 中删除 `title`，新增 `embed_text()`**

在 `src/brain/models.py` 中把 `PatternCard` 整体替换为：

```python
@dataclass
class PatternCard:
    id: str
    description: str
    template: str
    examples: list[str]
    frequency: FrequencyProfile
    source: str
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "template": self.template,
            "examples": self.examples,
            "frequency": {
                "recent": self.frequency.recent,
                "medium": self.frequency.medium,
                "long_term": self.frequency.long_term,
                "total": self.frequency.total,
            },
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PatternCard:
        freq = d["frequency"]
        return cls(
            id=d["id"],
            description=d["description"],
            template=d["template"],
            examples=d["examples"],
            frequency=FrequencyProfile(
                recent=freq["recent"],
                medium=freq["medium"],
                long_term=freq["long_term"],
                total=freq["total"],
            ),
            source=d["source"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )

    def embed_text(self) -> str:
        """语义检索用的 embedding 文本（description + examples）。"""
        examples_text = " / ".join(self.examples)
        return f"{self.description} 例句：{examples_text}"
```

- [ ] **Step 4: 更新 `config.py`**

在 `src/brain/config.py` 中把 `CHROMA_DIR` 改为 `LANCEDB_DIR`，并在末尾追加 `DEDUP_TOP_N`：

```python
LANCEDB_DIR = DATA_DIR / "lancedb"
```

删掉 `CHROMA_DIR = DATA_DIR / "chromadb"` 那行。

在文件末尾追加：

```python
DEDUP_TOP_N = int(os.environ.get("DEDUP_TOP_N", "3"))
```

- [ ] **Step 5: Commit**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
git add pyproject.toml src/brain/models.py src/brain/config.py uv.lock
git commit -m "refactor(brain): 替换 chromadb 依赖为 lancedb，删除 title 字段，更新配置"
```

---

## Task 2: LanceDB 存储层重写

**Files:**
- Rewrite: `src/brain/store/embedding.py`
- Rewrite: `src/brain/store/pattern_db.py`
- Modify: `src/brain/store/retriever.py`

- [ ] **Step 1: 重写 `embedding.py`**

把 `src/brain/store/embedding.py` 完整替换为：

```python
from __future__ import annotations

from openai import OpenAI

from brain.config import EMBED_API_BASE, EMBED_API_KEY, EMBED_MODEL, EMBED_DIMENSIONS


class QwenEmbedder:
    """封装 OpenAI 兼容的 embedding API，返回原始向量列表。"""

    def __init__(self):
        self._client = OpenAI(base_url=EMBED_API_BASE, api_key=EMBED_API_KEY)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            dimensions=EMBED_DIMENSIONS,
        )
        return [item.embedding for item in response.data]
```

- [ ] **Step 2: 重写 `pattern_db.py`**

把 `src/brain/store/pattern_db.py` 完整替换为：

```python
from __future__ import annotations

import json
from pathlib import Path

import lancedb
import pyarrow as pa

from brain.config import EMBED_DIMENSIONS
from brain.models import PatternCard


class PatternDB:
    TABLE_NAME = "patterns"

    def __init__(self, persist_dir: Path | str, embedder=None):
        self.persist_dir = Path(persist_dir)
        self._db = lancedb.connect(str(self.persist_dir))
        self._embedder = embedder
        self._table = None
        if self.TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(self.TABLE_NAME)

    def _make_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("json", pa.string()),
            pa.field("vec_template", pa.list_(pa.float32(), EMBED_DIMENSIONS)),
            pa.field("vec_semantic", pa.list_(pa.float32(), EMBED_DIMENSIONS)),
        ])

    def _cards_to_data(self, cards: list[PatternCard]) -> list[dict]:
        templates = [c.template for c in cards]
        semantics = [c.embed_text() for c in cards]
        vec_t = self._embedder.embed(templates)
        vec_s = self._embedder.embed(semantics)
        return [
            {
                "id": c.id,
                "json": json.dumps(c.to_dict(), ensure_ascii=False),
                "vec_template": vt,
                "vec_semantic": vs,
            }
            for c, vt, vs in zip(cards, vec_t, vec_s)
        ]

    def save(self, cards: list[PatternCard]) -> None:
        if not cards:
            return
        data = self._cards_to_data(cards)
        if self._table is None:
            self._table = self._db.create_table(
                self.TABLE_NAME, data, schema=self._make_schema(),
            )
        else:
            self._table.merge_insert("id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute(data)

    def update(self, cards: list[PatternCard]) -> None:
        self.save(cards)

    def get(self, pattern_id: str) -> PatternCard | None:
        if self._table is None:
            return None
        try:
            rows = self._table.search() \
                .where(f"id = '{pattern_id}'") \
                .limit(1) \
                .to_list()
            if not rows:
                return None
            return PatternCard.from_dict(json.loads(rows[0]["json"]))
        except Exception:
            return None

    def list_all(self) -> list[PatternCard]:
        if self._table is None:
            return []
        rows = self._table.to_pandas()
        return [
            PatternCard.from_dict(json.loads(row["json"]))
            for _, row in rows.iterrows()
        ]

    def count(self) -> int:
        if self._table is None:
            return 0
        return self._table.count_rows()

    def query_by_template(
        self, text: str, top_k: int = 3,
    ) -> list[tuple[PatternCard, float]]:
        """按 template 向量列检索，返回 (card, similarity) 列表。"""
        if self._table is None or self._table.count_rows() == 0:
            return []
        vec = self._embedder.embed([text])[0]
        n = min(top_k, self._table.count_rows())
        results = self._table.search(vec, vector_column_name="vec_template") \
            .metric("cosine") \
            .limit(n) \
            .to_list()
        return [
            (PatternCard.from_dict(json.loads(r["json"])), 1.0 - r["_distance"])
            for r in results
        ]

    def query_by_semantic(
        self, text: str, top_k: int = 8,
    ) -> list[tuple[PatternCard, float]]:
        """按 semantic 向量列检索（description + examples），返回 (card, similarity) 列表。"""
        if self._table is None or self._table.count_rows() == 0:
            return []
        vec = self._embedder.embed([text])[0]
        n = min(top_k, self._table.count_rows())
        results = self._table.search(vec, vector_column_name="vec_semantic") \
            .metric("cosine") \
            .limit(n) \
            .to_list()
        return [
            (PatternCard.from_dict(json.loads(r["json"])), 1.0 - r["_distance"])
            for r in results
        ]
```

- [ ] **Step 3: 更新 `retriever.py`**

把 `src/brain/store/retriever.py` 完整替换为：

```python
from __future__ import annotations

from brain.config import RETRIEVAL_TOP_K, SIMILARITY_WEIGHT, FRESHNESS_WEIGHT
from brain.models import PatternCard
from brain.store.pattern_db import PatternDB


def retrieve_patterns(
    db: PatternDB,
    conversation_text: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[PatternCard]:
    raw_results = db.query_by_semantic(conversation_text, top_k=top_k * 2)
    if not raw_results:
        return []

    scored = [
        (card, similarity * SIMILARITY_WEIGHT + card.frequency.freshness * FRESHNESS_WEIGHT)
        for card, similarity in raw_results
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [card for card, _ in scored[:top_k]]
```

- [ ] **Step 4: Commit**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
git add src/brain/store/
git commit -m "feat(brain): 重写存储层为 LanceDB，支持双向量列（vec_template + vec_semantic）"
```

---

## Task 3: 消费层适配（删除 title 引用）

**Files:**
- Modify: `src/brain/extract/refiner.py`
- Modify: `src/brain/compose/menu.py`
- Modify: `src/brain/compose/tools.py`
- Modify: `src/brain/viewer.py`
- Modify: `scripts/streamlit_patterns.py`

- [ ] **Step 1: 更新 extraction prompt 和 `extract_from_chunk`**

在 `src/brain/extract/refiner.py` 中把 `_EXTRACT_PROMPT` 替换为：

```python
_EXTRACT_PROMPT = """\
以下是 B 站某视频下的用户评论。请从中发现值得收录的语言模式——特别是：
- 多人使用的相似句式
- 不像 AI 会自然生成的表达
- 可以替换内容复用的句式模板

对每个发现的模式，输出 JSON 数组，每个元素包含：
{
  "template": "含 [A] [B] 占位符的模板句",
  "examples": ["2-5个真实例句"],
  "description": "模式描述：是什么、什么时候用、传达什么感觉"
}

如果这批评论中没有值得收录的模式，返回空数组 []。
只输出 JSON，不要其他文字。

评论：
"""
```

在 `extract_from_chunk` 函数中，把字段校验行和 `PatternCard` 构造改为（删掉 title）：

```python
    for p in raw_patterns:
        if not all(k in p for k in ("template", "examples", "description")):
            continue
        cards.append(
            PatternCard(
                id=f"pat-{uuid.uuid4().hex[:8]}",
                description=p["description"],
                template=p["template"],
                examples=p["examples"][:5],
                frequency=FrequencyProfile(recent=1, medium=1, long_term=1, total=1),
                source="bilibili",
                created_at=now,
                updated_at=now,
            )
        )
```

- [ ] **Step 2: 更新 `compose/menu.py`**

把 `build_menu` 中 `lines.append(...)` 行改为：

```python
        lines.append(f'{i}. [{p.id}] "{p.template}" — {p.description[:30]}')
```

- [ ] **Step 3: 更新 `compose/tools.py`**

把 `handle_inspect_pattern` 的返回文本改为（删掉标题行）：

```python
    examples_text = "\n".join(f"  - {ex}" for ex in card.examples)
    return (
        f"模板：{card.template}\n"
        f"描述：{card.description}\n"
        f"例句：\n{examples_text}"
    )
```

- [ ] **Step 4: 更新 `viewer.py`**

把 `_SORT_OPTIONS` 改为：

```python
_SORT_OPTIONS = {"updated_at", "freshness", "template"}
```

把 `filter_patterns` 中的 haystack 改为（删掉 `card.title`）：

```python
        haystack = "\n".join(
            [
                card.template,
                card.description,
                *card.examples,
            ]
        ).lower()
```

把 `sort_patterns` 中的 `title` 分支改为 `template`：

```python
    if sort_by == "template":
        return sorted(patterns, key=lambda card: card.template)
```

把 `format_pattern_summary` 改为：

```python
def format_pattern_summary(card: PatternCard) -> str:
    return (
        f"{card.template} | "
        f"freshness={card.frequency.freshness:.2f} | total={card.frequency.total}"
    )
```

- [ ] **Step 5: 更新 `scripts/streamlit_patterns.py`**

把 `render_pattern_card` 中 `st.markdown(f"**标题**: {card.title}")` 那一行删掉。

把 `from brain.config import CHROMA_DIR, RETRIEVAL_TOP_K` 改为：

```python
from brain.config import LANCEDB_DIR, RETRIEVAL_TOP_K
```

把 `get_db()` 中 `PatternDB(CHROMA_DIR)` 改为 `PatternDB(LANCEDB_DIR)`。

把排序选项改为：

```python
        sort_by = st.selectbox(
            "排序方式",
            options=["updated_at", "freshness", "template"],
            format_func=lambda x: {
                "updated_at": "最近更新",
                "freshness": "热度 freshness",
                "template": "模板",
            }[x],
        )
```

把搜索框 placeholder 改为：

```python
            placeholder="按模板、描述或例句搜索",
```

- [ ] **Step 6: Commit**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
git add src/brain/extract/refiner.py src/brain/compose/ src/brain/viewer.py scripts/streamlit_patterns.py
git commit -m "refactor(brain): 全面移除 title 引用，展示改用 template，streamlit 适配 LanceDB"
```

---

## Task 4: 修复所有测试

**Files:**
- Modify: `tests/test_extract.py`
- Modify: `tests/test_compose.py`
- Rewrite: `tests/test_store.py`

- [ ] **Step 1: 创建 `tests/conftest.py` 提供 `MockEmbedder`**

创建 `tests/conftest.py`：

```python
from __future__ import annotations

import hashlib
import random as _random


class MockEmbedder:
    """确定性 mock embedder：相同文本 → 相同单位向量（cos=1.0）；不同文本 → 近似正交。"""
    DIM = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
            rng = _random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self.DIM)]
            norm = sum(x * x for x in vec) ** 0.5
            result.append([x / norm for x in vec])
        return result
```

- [ ] **Step 2: 修复 `tests/test_extract.py`**

把文件完整替换为：

```python
import json
import os
from unittest.mock import patch
from datetime import datetime

import pytest

from brain.models import CleanedComment, PatternCard, FrequencyProfile
from brain.extract.chunker import chunk_comments
from brain.extract.refiner import extract_from_chunk, deduplicate_and_merge


def _make_comment(rpid: int, message: str) -> CleanedComment:
    return CleanedComment(rpid=rpid, bvid="BV1test", uid=1, message=message, ctime=1700000000)


class TestChunker:
    def test_basic_chunking(self):
        comments = [_make_comment(i, f"comment number {i}") for i in range(120)]
        chunks = chunk_comments(comments, chunk_size=50)
        assert len(chunks) == 3
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50
        assert len(chunks[2]) == 20

    def test_small_input(self):
        comments = [_make_comment(i, f"comment {i}") for i in range(10)]
        chunks = chunk_comments(comments, chunk_size=50)
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_empty_input(self):
        chunks = chunk_comments([], chunk_size=50)
        assert chunks == []

    def test_returns_message_strings(self):
        comments = [_make_comment(1, "hello world")]
        chunks = chunk_comments(comments, chunk_size=50)
        assert chunks == [["hello world"]]


class TestExtractFromChunk:
    def test_parses_llm_response(self):
        mock_patterns = [
            {
                "template": "[A]...好家伙...",
                "examples": ["这也行...好家伙...", "又来了...好家伙..."],
                "description": "表达无奈和吐槽",
            }
        ]
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=(json.dumps(mock_patterns, ensure_ascii=False), 10, 50),
        ):
            cards, total_tokens = extract_from_chunk(["comment1", "comment2"])

        assert len(cards) == 1
        assert cards[0].template == "[A]...好家伙..."

    def test_handles_empty_response(self):
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=("[]", 5, 3),
        ):
            cards, total_tokens = extract_from_chunk(["comment1", "comment2"])

        assert cards == []
        assert total_tokens == 8


class TestDeduplicateAndMerge:
    pass  # 将在 Task 8 中重写


@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set — skipping integration test",
)
class TestExtractIntegration:
    def test_real_extraction(self):
        """Smoke test: send real comments to LLM and check output structure."""
        comments = [
            "这也行...好家伙...",
            "又来了...好家伙...",
            "绝了...好家伙...",
            "笑死我了",
            "前方高能",
            "你说得对，但是这个视频确实不错",
            "up主下次还敢",
            "建议下次不要建议了",
            "太真实了",
            "我直接好家伙",
        ]
        cards, _ = extract_from_chunk(comments)
        assert isinstance(cards, list)
        if cards:
            card = cards[0]
            assert card.template
            assert len(card.examples) > 0
```

- [ ] **Step 3: 修复 `tests/test_compose.py`**

把 `tests/test_compose.py` 完整替换为：

```python
from datetime import datetime

from brain.models import PatternCard, FrequencyProfile
from brain.store.pattern_db import PatternDB
from brain.compose.menu import build_menu
from brain.compose.tools import get_tool_definition, handle_inspect_pattern
from brain.compose.assembler import assemble_system_prompt

from conftest import MockEmbedder


def _make_card(id: str, template: str, description: str = "test desc") -> PatternCard:
    return PatternCard(
        id=id,
        description=description,
        template=template,
        examples=[f"example of {template}"],
        frequency=FrequencyProfile(recent=5, medium=10, long_term=30, total=45),
        source="test",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


class TestBuildMenu:
    def test_formats_pattern_list(self):
        cards = [
            _make_card("pat-001", "[A]...好家伙..."),
            _make_card("pat-002", "[A]嘛~[B]啦~"),
        ]
        menu = build_menu(cards)
        assert "[pat-001]" in menu
        assert "[A]...好家伙..." in menu
        assert "[pat-002]" in menu

    def test_empty_patterns(self):
        menu = build_menu([])
        assert "没有" in menu or "无" in menu or menu.strip() == ""


class TestToolDefinition:
    def test_tool_schema(self):
        tool = get_tool_definition()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "inspect_pattern"
        assert "pattern_id" in tool["function"]["parameters"]["properties"]


class TestHandleInspectPattern:
    def test_returns_pattern_details(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        card = _make_card("pat-001", "[A]...好家伙...")
        db.save([card])
        result = handle_inspect_pattern(db, "pat-001")
        assert "[A]...好家伙..." in result
        assert "desc" in result

    def test_pattern_not_found(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        result = handle_inspect_pattern(db, "nonexistent")
        assert "找不到" in result or "not found" in result.lower()


class TestAssembler:
    def test_assembles_with_patterns(self):
        cards = [
            _make_card("pat-001", "[A]...好家伙..."),
            _make_card("pat-002", "[A]嘛~[B]啦~"),
        ]
        prompt = assemble_system_prompt(cards)
        assert "[A]...好家伙..." in prompt
        assert "[A]嘛~[B]啦~" in prompt
        assert "inspect_pattern" in prompt or "查看" in prompt

    def test_assembles_without_patterns(self):
        prompt = assemble_system_prompt([])
        assert isinstance(prompt, str)
        assert len(prompt) > 0
```

- [ ] **Step 4: 重写 `tests/test_store.py`**

把 `tests/test_store.py` 完整替换为：

```python
from datetime import datetime
from unittest.mock import patch, MagicMock

from brain.models import PatternCard, FrequencyProfile
from brain.store.pattern_db import PatternDB
from brain.store.retriever import retrieve_patterns
from brain.store.embedding import QwenEmbedder

from conftest import MockEmbedder


def _make_card(
    id: str, template: str = "[A] test", description: str = "test desc",
) -> PatternCard:
    return PatternCard(
        id=id,
        description=description,
        template=template,
        examples=[f"example of {template}"],
        frequency=FrequencyProfile(recent=5, medium=10, long_term=30, total=45),
        source="test",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


class TestPatternDB:
    def test_save_and_get(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        card = _make_card("pat-001", "[A] test pattern")
        db.save([card])
        retrieved = db.get("pat-001")
        assert retrieved is not None
        assert retrieved.template == "[A] test pattern"
        assert retrieved.frequency.recent == 5

    def test_list_all(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([_make_card("1", "[A] alpha"), _make_card("2", "[A] beta")])
        all_cards = db.list_all()
        assert len(all_cards) == 2

    def test_update(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        card = _make_card("pat-001", "[A] test pattern")
        db.save([card])
        card.frequency.recent = 99
        card.examples.append("new example")
        db.update([card])
        retrieved = db.get("pat-001")
        assert retrieved.frequency.recent == 99
        assert "new example" in retrieved.examples

    def test_save_empty_list(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([])
        assert db.list_all() == []

    def test_count(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        assert db.count() == 0
        db.save([_make_card("1"), _make_card("2")])
        assert db.count() == 2

    def test_query_by_template(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([
            _make_card("1", "[A]太离谱了"),
            _make_card("2", "[A]嘛~[B]啦~"),
        ])
        results = db.query_by_template("[A]太离谱了", top_k=1)
        assert len(results) == 1
        card, sim = results[0]
        assert isinstance(sim, float)

    def test_query_by_semantic(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([
            _make_card("1", "[A]太离谱了", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "[A]嘛~[B]啦~", "用叠字和语气词表达撒娇"),
        ])
        results = db.query_by_semantic("对离谱事情表达无奈的吐槽", top_k=1)
        assert len(results) == 1

    def test_query_empty_db(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        assert db.query_by_template("test") == []
        assert db.query_by_semantic("test") == []


class TestRetriever:
    def test_retrieve_returns_cards(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([
            _make_card("1", "[A]太离谱了", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "[A]嘛~[B]啦~", "用叠字和语气词表达撒娇"),
            _make_card("3", "[A]难道不是[B]吗", "用反问表达不满"),
        ])
        results = retrieve_patterns(db, "这也太离谱了吧", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(c, PatternCard) for c in results)

    def test_retrieve_empty_db(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        results = retrieve_patterns(db, "test query", top_k=5)
        assert results == []

    def test_retrieve_respects_top_k(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([
            _make_card("1", "[A]太离谱了", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "[A]嘛~[B]啦~", "用叠字和语气词表达撒娇"),
            _make_card("3", "[A]难道不是[B]吗", "用反问表达不满"),
        ])
        results = retrieve_patterns(db, "这也太离谱了吧", top_k=1)
        assert len(results) == 1


class TestQwenEmbedder:
    def test_embed_returns_vectors(self):
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 2048),
            MagicMock(embedding=[0.2] * 2048),
        ]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("brain.store.embedding.OpenAI", return_value=mock_client):
            embedder = QwenEmbedder()
            result = embedder.embed(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 2048

    def test_embed_empty_list(self):
        embedder = QwenEmbedder.__new__(QwenEmbedder)
        assert embedder.embed([]) == []
```

- [ ] **Step 5: 运行全部测试确认通过**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/ -v
```

期望: 全部 PASS（`TestDeduplicateAndMerge` 暂时是空壳 pass）。

- [ ] **Step 6: Commit**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
git add tests/
git commit -m "test(brain): 适配 LanceDB + 删除 title，引入 MockEmbedder"
```

---

## Task 5: LLM 1-vs-N 判重函数 `_judge_duplicate_topn`

**Files:**
- Modify: `src/brain/extract/refiner.py`
- Modify: `tests/test_extract.py`

- [ ] **Step 1: 在 `tests/test_extract.py` 中添加 `TestJudgeDuplicateTopN` 类（先失败）**

在顶部 import 补上：

```python
from brain.extract.refiner import extract_from_chunk, deduplicate_and_merge, _judge_duplicate_topn
```

在 `TestDeduplicateAndMerge` 之后添加：

```python
class TestJudgeDuplicateTopN:
    def _make_card(self, cid: str, template: str, description: str = "默认描述") -> PatternCard:
        return PatternCard(
            id=cid, description=description,
            template=template, examples=["示例"],
            frequency=FrequencyProfile(1, 1, 1, 1),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_finds_duplicate_in_candidates(self):
        resp = '{"duplicate_of": 1, "keep_description": "candidate", "reason": "same"}'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, keep = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]好家伙[B]"), self._make_card("c2", "[A]绝了")],
            )
        assert idx == 0
        assert keep == "candidate"

    def test_no_duplicate_returns_none(self):
        resp = '{"duplicate_of": 0, "keep_description": "current", "reason": "all different"}'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, keep = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了")],
            )
        assert idx is None

    def test_parse_error_returns_none(self):
        with patch("brain.extract.refiner._call_llm_streaming", return_value=("bad json", 5, 5)):
            idx, keep = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了")],
            )
        assert idx is None
        assert keep == "current"

    def test_handles_markdown_code_fence(self):
        resp = '```json\n{"duplicate_of": 2, "keep_description": "current", "reason": "same"}\n```'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, keep = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了"), self._make_card("c2", "[A]好家伙[B]")],
            )
        assert idx == 1
        assert keep == "current"

    def test_empty_candidates_returns_none(self):
        idx, keep = _judge_duplicate_topn(
            self._make_card("new", "[A]好家伙"),
            [],
        )
        assert idx is None
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/test_extract.py::TestJudgeDuplicateTopN -v
```

期望: `ImportError: cannot import name '_judge_duplicate_topn'`

- [ ] **Step 3: 在 `refiner.py` 中添加 prompt + 函数**

在 `_EXTRACT_PROMPT` 之后（`_call_llm_streaming` 之前）插入：

```python
_DEDUP_JUDGE_PROMPT = """\
以下是一个从 B 站评论中提取的语言模式（当前模式），以及若干个从数据库中检索到的相似候选。
请判断候选中是否有与当前模式描述同一种语言模式的条目（语义等价或高度相似，可以合并为一条记录）。

【当前模式】
模板: {current_template}
描述: {current_desc}
例句: {current_examples}

【相似候选】
{candidates_block}

如果候选中有重复的，请选出最佳匹配（只选一个）。如果都不重复，输出 0。

输出 JSON:
{{
  "duplicate_of": 0,
  "keep_description": "current" 或 "candidate",
  "reason": "一句话说明"
}}

duplicate_of: 候选编号（1 开始），0 表示无重复。
keep_description: 哪一方的描述更完整准确，仅在有重复时有意义。
只输出 JSON，不要其他文字。
"""


def _judge_duplicate_topn(
    card: PatternCard,
    candidates: list[PatternCard],
) -> tuple[int | None, str]:
    """询问 LLM 候选中是否有和 card 重复的模式。
    返回 (candidate_index_0based | None, keep_description: 'current'|'candidate')。
    解析失败或无重复返回 (None, 'current')。
    """
    if not candidates:
        return None, "current"

    parts = []
    for i, c in enumerate(candidates, 1):
        parts.append(
            f"--- 候选 {i} ({c.id}) ---\n"
            f"模板: {c.template}\n"
            f"描述: {c.description}\n"
            f"例句: {' / '.join(c.examples[:3])}"
        )
    candidates_block = "\n".join(parts)

    prompt = _DEDUP_JUDGE_PROMPT.format(
        current_template=card.template,
        current_desc=card.description,
        current_examples=" / ".join(card.examples[:3]),
        candidates_block=candidates_block,
    )
    content, _, _ = _call_llm_streaming(prompt)
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    try:
        result = json.loads(content)
        dup_of = int(result.get("duplicate_of", 0))
        keep = str(result.get("keep_description", "current"))
        if dup_of < 1 or dup_of > len(candidates):
            return None, "current"
        return dup_of - 1, keep
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, "current"
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/test_extract.py::TestJudgeDuplicateTopN -v
```

期望: 5 个测试全 PASS。

- [ ] **Step 5: Commit**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
git add src/brain/extract/refiner.py tests/test_extract.py
git commit -m "feat(brain): 添加 1-vs-N LLM 判重函数 _judge_duplicate_topn"
```

---

## Task 6: 批次内去重 `_dedup_intra_batch`

**Files:**
- Modify: `src/brain/extract/refiner.py`
- Modify: `tests/test_extract.py`

- [ ] **Step 1: 在 `tests/test_extract.py` 中添加 import 和 `TestIntraBatchDedup` 类（先失败）**

在顶部 import 补上：

```python
from brain.extract.refiner import _dedup_intra_batch
from brain.store.pattern_db import PatternDB
from conftest import MockEmbedder
```

在 `TestJudgeDuplicateTopN` 之后添加：

```python
class TestIntraBatchDedup:
    def _make_card(self, cid: str, template: str, description: str = "默认描述",
                   examples: list[str] | None = None) -> PatternCard:
        return PatternCard(
            id=cid, description=description,
            template=template, examples=examples or ["示例"],
            frequency=FrequencyProfile(1, 1, 1, 1),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_single_card_passes_through(self):
        embedder = MockEmbedder()
        card = self._make_card("1", "[A]模式")
        result = _dedup_intra_batch([card], embedder, top_n=3)
        assert len(result) == 1
        assert result[0].id == "1"

    def test_different_cards_both_kept(self):
        embedder = MockEmbedder()
        card_a = self._make_card("1", "[A]完全不同A", description="descA", examples=["exA"])
        card_b = self._make_card("2", "[A]完全不同B", description="descB", examples=["exB"])
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, "current")):
            result = _dedup_intra_batch([card_a, card_b], embedder, top_n=3)
        assert len(result) == 2

    def test_duplicate_cards_merged(self):
        embedder = MockEmbedder()
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description=same_desc, examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "current")):
            result = _dedup_intra_batch([card_a, card_b], embedder, top_n=3)
        assert len(result) == 1
        assert result[0].id == "1"
        assert result[0].frequency.total == 2

    def test_description_updated_when_keep_current(self):
        embedder = MockEmbedder()
        same_tmpl, same_ex = "[A]好家伙", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description="旧描述", examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description="旧描述", examples=same_ex)
        card_b.description = "更好的描述"
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "current")):
            result = _dedup_intra_batch([card_a, card_b], embedder, top_n=3)
        assert len(result) == 1
        assert result[0].description == "更好的描述"

    def test_non_duplicate_keeps_both(self):
        embedder = MockEmbedder()
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description=same_desc, examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, "current")):
            result = _dedup_intra_batch([card_a, card_b], embedder, top_n=3)
        assert len(result) == 2
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/test_extract.py::TestIntraBatchDedup -v
```

期望: `ImportError: cannot import name '_dedup_intra_batch'`

- [ ] **Step 3: 在 `refiner.py` 中添加依赖 import 和 `_dedup_intra_batch`**

在 `refiner.py` 顶部 import 区域中，`from openai import OpenAI` 之后添加：

```python
import tempfile

import lancedb
import pyarrow as pa
from rich.console import Console
```

将 `from brain.config import ...` 改为：

```python
from brain.config import DATA_DIR, LLM_API_BASE, LLM_API_KEY, LLM_MODEL, DEDUP_TOP_N, EMBED_DIMENSIONS
from brain.store.pattern_db import PatternDB
```

在 `_client = OpenAI(...)` 之后添加：

```python
_console = Console()
```

然后在 `_judge_duplicate_topn` 之后、`extract_from_chunk` 之前插入：

```python
def _merge_hits(
    hits_a: list[tuple[PatternCard, float]],
    hits_b: list[tuple[PatternCard, float]],
) -> list[PatternCard]:
    """合并两路检索结果，按 card.id 去重。"""
    seen: dict[str, PatternCard] = {}
    for card, _ in hits_a + hits_b:
        if card.id not in seen:
            seen[card.id] = card
    return list(seen.values())


def _dedup_intra_batch(
    cards: list[PatternCard],
    embedder,
    top_n: int = DEDUP_TOP_N,
) -> list[PatternCard]:
    """批次内去重：用临时 LanceDB 表做双路检索（vec_template + vec_semantic），
    合并候选后交 LLM 判重。日志并列输出两路结果。
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = lancedb.connect(tmp_dir)
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("json", pa.string()),
            pa.field("vec_template", pa.list_(pa.float32(), EMBED_DIMENSIONS)),
            pa.field("vec_semantic", pa.list_(pa.float32(), EMBED_DIMENSIONS)),
        ])
        tmp_table = None
        kept: dict[str, PatternCard] = {}

        for card in cards:
            vec_t = embedder.embed([card.template])[0]
            vec_s = embedder.embed([card.embed_text()])[0]
            row = {
                "id": card.id,
                "json": json.dumps(card.to_dict(), ensure_ascii=False),
                "vec_template": vec_t,
                "vec_semantic": vec_s,
            }

            if tmp_table is None:
                tmp_table = tmp_db.create_table("batch", [row], schema=schema)
                kept[card.id] = card
                continue

            n = min(top_n, tmp_table.count_rows())

            # 双路检索
            hits_tmpl = [
                (PatternCard.from_dict(json.loads(r["json"])), 1.0 - r["_distance"])
                for r in tmp_table.search(vec_t, vector_column_name="vec_template")
                    .metric("cosine").limit(n).to_list()
            ]
            hits_sem = [
                (PatternCard.from_dict(json.loads(r["json"])), 1.0 - r["_distance"])
                for r in tmp_table.search(vec_s, vector_column_name="vec_semantic")
                    .metric("cosine").limit(n).to_list()
            ]
            merged = _merge_hits(hits_tmpl, hits_sem)

            tmpl_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_tmpl)
            sem_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_sem)
            _console.print(f"批次内  [cyan]{card.template!r}[/cyan]")
            _console.print(f"  vec_template: {tmpl_str}")
            _console.print(f"  vec_semantic: {sem_str}")
            _console.print(f"  合并候选(去重): {len(merged)}个 → LLM 判断")

            dup_idx, keep_desc = _judge_duplicate_topn(card, merged)
            if dup_idx is not None:
                matched = merged[dup_idx]
                target = kept[matched.id]
                if keep_desc == "current":
                    target.description = card.description
                _merge_into(target, card)
                # 更新临时表中的记录
                new_vec_s = embedder.embed([target.embed_text()])[0]
                tmp_table.merge_insert("id") \
                    .when_matched_update_all() \
                    .when_not_matched_insert_all() \
                    .execute([{
                        "id": target.id,
                        "json": json.dumps(target.to_dict(), ensure_ascii=False),
                        "vec_template": embedder.embed([target.template])[0],
                        "vec_semantic": new_vec_s,
                    }])
                _console.print(
                    f"  [green]→ LLM: 与 {matched.template!r} 重复"
                    f" · 保留描述={'当前' if keep_desc == 'current' else '候选'}"
                    f" · 合并[/green]"
                )
                continue

            _console.print(f"  [dim]→ LLM: 无重复，保留[/dim]")
            tmp_table.add([row])
            kept[card.id] = card

    return list(kept.values())
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/test_extract.py::TestIntraBatchDedup -v
```

期望: 5 个测试全 PASS。

- [ ] **Step 5: Commit**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
git add src/brain/extract/refiner.py tests/test_extract.py
git commit -m "feat(brain): 实现批次内 LanceDB 双路去重 _dedup_intra_batch"
```

---

## Task 7: 入库前去重 `_dedup_against_db`

**Files:**
- Modify: `src/brain/extract/refiner.py`
- Modify: `tests/test_extract.py`

- [ ] **Step 1: 在 `tests/test_extract.py` 中添加 `TestCrossDbDedup` 类（先失败）**

更新 import 行，加入 `_dedup_against_db`：

```python
from brain.extract.refiner import (
    extract_from_chunk, deduplicate_and_merge,
    _judge_duplicate_topn, _dedup_intra_batch, _dedup_against_db,
)
```

在 `TestIntraBatchDedup` 之后添加：

```python
class TestCrossDbDedup:
    def _make_card(self, cid: str, template: str, description: str = "默认描述",
                   examples: list[str] | None = None) -> PatternCard:
        return PatternCard(
            id=cid, description=description,
            template=template, examples=examples or ["示例"],
            frequency=FrequencyProfile(1, 1, 1, 1),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_empty_db_returns_all_as_new(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        cards = [self._make_card("1", "[A]模式A"), self._make_card("2", "[A]模式B")]
        new, updates = _dedup_against_db(cards, db, top_n=3)
        assert len(new) == 2
        assert len(updates) == 0

    def test_similar_to_existing_triggers_llm_and_updates(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(5, 5, 5, 5)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "candidate")):
            new, updates = _dedup_against_db([new_card], db, top_n=3)

        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].id == "existing-1"
        assert updates[0].frequency.total == 6

    def test_different_from_existing_added_as_new(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        existing = self._make_card("existing-1", "[A]完全不同A", description="descA", examples=["exA"])
        db.save([existing])

        new_card = self._make_card("new-1", "[A]完全不同B", description="descB", examples=["exB"])
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, "current")):
            new, updates = _dedup_against_db([new_card], db, top_n=3)

        assert len(new) == 1
        assert len(updates) == 0

    def test_description_updated_when_keep_current(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description="旧描述", examples=same_ex)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description="更好的新描述", examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "current")):
            new, updates = _dedup_against_db([new_card], db, top_n=3)

        assert len(updates) == 1
        assert updates[0].description == "更好的新描述"

    def test_two_new_cards_merge_into_same_existing(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(5, 5, 5, 5)
        db.save([existing])

        card_x = self._make_card("x", same_tmpl, description=same_desc, examples=same_ex)
        card_y = self._make_card("y", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "candidate")):
            new, updates = _dedup_against_db([card_x, card_y], db, top_n=3)

        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].frequency.total == 7
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/test_extract.py::TestCrossDbDedup -v
```

期望: `ImportError: cannot import name '_dedup_against_db'`

- [ ] **Step 3: 在 `refiner.py` 中添加 `_dedup_against_db`**

在 `_dedup_intra_batch` 之后、`extract_from_chunk` 之前插入：

```python
def _dedup_against_db(
    cards: list[PatternCard],
    db: PatternDB,
    top_n: int = DEDUP_TOP_N,
) -> tuple[list[PatternCard], list[PatternCard]]:
    """入库前去重：每张新卡片对 DB 做双路检索（vec_template + vec_semantic），
    合并候选后交 LLM 判重。若多张新卡片都与同一已有卡片重复，合并到同一对象上。
    日志并列输出两路结果。
    """
    new_cards: list[PatternCard] = []
    updates_by_id: dict[str, PatternCard] = {}

    for card in cards:
        hits_tmpl = db.query_by_template(card.template, top_k=top_n)
        hits_sem = db.query_by_semantic(card.embed_text(), top_k=top_n)

        if not hits_tmpl and not hits_sem:
            new_cards.append(card)
            continue

        merged = _merge_hits(hits_tmpl, hits_sem)

        tmpl_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_tmpl) or "(空)"
        sem_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_sem) or "(空)"
        _console.print(f"入库去重  [cyan]{card.template!r}[/cyan]")
        _console.print(f"  vec_template: {tmpl_str}")
        _console.print(f"  vec_semantic: {sem_str}")
        _console.print(f"  合并候选(去重): {len(merged)}个 → LLM 判断")

        dup_idx, keep_desc = _judge_duplicate_topn(card, merged)
        if dup_idx is not None:
            matched = merged[dup_idx]
            target = updates_by_id.get(matched.id, matched)
            if keep_desc == "current":
                target.description = card.description
            _merge_into(target, card)
            updates_by_id[target.id] = target
            _console.print(
                f"  [green]→ LLM: 与 {matched.template!r} ({matched.id}) 重复"
                f" · 保留描述={'当前' if keep_desc == 'current' else '候选'}"
                f" · 更新已有[/green]"
            )
            continue

        _console.print(f"  [dim]→ LLM: 无重复，新增[/dim]")
        new_cards.append(card)

    return new_cards, list(updates_by_id.values())
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/test_extract.py::TestCrossDbDedup -v
```

期望: 5 个测试全 PASS。

- [ ] **Step 5: Commit**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
git add src/brain/extract/refiner.py tests/test_extract.py
git commit -m "feat(brain): 实现入库前 LanceDB 双路去重 _dedup_against_db"
```

---

## Task 8: 替换 `deduplicate_and_merge` + 更新 pipeline

**Files:**
- Modify: `src/brain/extract/refiner.py`
- Modify: `scripts/run_pipeline.py`
- Modify: `tests/test_extract.py`

- [ ] **Step 1: 重写 `TestDeduplicateAndMerge`**

把 `tests/test_extract.py` 中 `TestDeduplicateAndMerge` 替换为：

```python
class TestDeduplicateAndMerge:
    def _make_card(self, cid: str, template: str, description: str = "默认",
                   examples: list[str] | None = None) -> PatternCard:
        return PatternCard(
            id=cid, description=description,
            template=template, examples=examples or ["示例"],
            frequency=FrequencyProfile(1, 1, 1, 1),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_no_duplicates(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        cards = [
            self._make_card("1", "[A]不同A", description="descA", examples=["exA"]),
            self._make_card("2", "[A]不同B", description="descB", examples=["exB"]),
        ]
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, "current")):
            new, updates = deduplicate_and_merge(cards, db, embedder)
        assert len(new) == 2
        assert len(updates) == 0

    def test_intra_batch_dedup(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽", ["好家伙"]
        cards = [
            self._make_card("1", same_tmpl, description=same_desc, examples=same_ex),
            self._make_card("2", same_tmpl, description=same_desc, examples=same_ex),
        ]
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "candidate")):
            new, updates = deduplicate_and_merge(cards, db, embedder)
        assert len(new) == 1
        assert len(updates) == 0

    def test_cross_db_dedup(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(10, 10, 10, 10)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "candidate")):
            new, updates = deduplicate_and_merge([new_card], db, embedder)
        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].id == "existing-1"
        assert updates[0].frequency.total == 11
```

- [ ] **Step 2: 替换 `refiner.py` 中旧 `deduplicate_and_merge`**

删掉旧的 `deduplicate_and_merge` 函数（以及已不需要的 `existing_by_title` 逻辑），替换为：

```python
def deduplicate_and_merge(
    cards: list[PatternCard],
    db: PatternDB,
    embedder,
    top_n: int = DEDUP_TOP_N,
) -> tuple[list[PatternCard], list[PatternCard]]:
    """两阶段去重：批次内相互去重 → 入库前与 DB 比对。
    使用 LanceDB 双向量列（vec_template + vec_semantic）做双路检索。
    返回 (new_cards, updated_existing_cards)。
    """
    _console.print(f"\n[bold]开始去重[/bold]: {len(cards)} 个待入库模式，top_n={top_n}")

    deduped = _dedup_intra_batch(cards, embedder, top_n)
    _console.print(f"\n批次内去重完成: {len(cards)} → [bold]{len(deduped)}[/bold] 个\n")

    new_cards, updates = _dedup_against_db(deduped, db, top_n)
    _console.print(
        f"\n去重完成: 新增 [green]{len(new_cards)}[/green] 个,"
        f" 更新 [yellow]{len(updates)}[/yellow] 个"
    )

    return new_cards, updates
```

保留 `_merge_into` 函数不变。

- [ ] **Step 3: 运行 `TestDeduplicateAndMerge` 确认通过**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/test_extract.py::TestDeduplicateAndMerge -v
```

期望: 3 个测试全 PASS。

- [ ] **Step 4: 更新 `scripts/run_pipeline.py`**

把 import 区域的：

```python
from brain.config import BILIBILI_DB_PATH, CHROMA_DIR, CHUNK_SIZE, STATE_FILE
```

改为：

```python
from brain.config import BILIBILI_DB_PATH, LANCEDB_DIR, CHUNK_SIZE, STATE_FILE
```

把：

```python
from brain.store.embedding import QwenEmbeddingFunction
```

改为：

```python
from brain.store.embedding import QwenEmbedder
```

把初始化部分改为：

```python
    embedder = QwenEmbedder()
    db = PatternDB(LANCEDB_DIR, embedder=embedder)
```

把去重部分改为：

```python
    # 4. 去重合并
    print(f"\n提取到的原始模式: {len(all_patterns)} 个")
    print(f"数据库中已有模式: {db.count()} 个")

    new_cards, updates = deduplicate_and_merge(all_patterns, db, embedder)

    db.save(new_cards)
    db.update(updates)
```

把最后的统计改为：

```python
    total = db.count()
    print(f"完成。数据库中模式总数: {total}")
```

- [ ] **Step 5: 运行全部测试最终确认**

```bash
cd /home/sparidae/projects/silicachan/solution-brain && uv run pytest tests/ -v
```

期望: 全部 PASS。

- [ ] **Step 6: 删除旧 ChromaDB 数据目录（手动提示）**

运行后提示用户删除 `data/chromadb/` 并用 `--full` 重跑 pipeline 重建 LanceDB 数据：

```bash
rm -rf data/chromadb/
uv run python scripts/run_pipeline.py --full
```

- [ ] **Step 7: Commit**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
git add src/brain/extract/refiner.py scripts/run_pipeline.py tests/test_extract.py
git commit -m "feat(brain): 替换 deduplicate_and_merge 为 LanceDB 双路去重，更新 pipeline"
```

---

## 运行时输出示例

pipeline 运行后，去重阶段终端输出如下（两路来自不同向量列，天然对齐）：

```
开始去重: 12 个待入库模式，top_n=3

批次内  '[A]...好家伙...'
  vec_template: '[A]好家伙[B]'(0.941), '[A]绝了'(0.182)
  vec_semantic: '[A]好家伙[B]'(0.873), '[A]爱了'(0.431), '[A]绝了'(0.312)
  合并候选(去重): 3个 → LLM 判断
  → LLM: 与 '[A]好家伙[B]' 重复 · 保留描述=当前 · 合并
批次内  '[A]爱了爱了'
  vec_template: '[A]...好家伙...'(0.182), '[A]绝了'(0.123)
  vec_semantic: '[A]绝了'(0.210), '[A]...好家伙...'(0.150)
  合并候选(去重): 2个 → LLM 判断
  → LLM: 无重复，保留

批次内去重完成: 12 → 9 个

入库去重  '[A]爱了爱了'
  vec_template: '[A]爱了[B]'(0.912), '[A]绝了'(0.430), '[A]好家伙'(0.211)
  vec_semantic: '[A]爱了[B]'(0.870), '[A]好家伙'(0.350), '[A]绝了'(0.310)
  合并候选(去重): 3个 → LLM 判断
  → LLM: 与 '[A]爱了[B]' (pat-abc1) 重复 · 保留描述=当前 · 更新已有
入库去重  '[A]太真实了'
  vec_template: '[A]真实'(0.430), '[A]好家伙'(0.211), '[A]绝了'(0.190)
  vec_semantic: '[A]真实'(0.510), '[A]绝了'(0.230), '[A]好家伙'(0.180)
  合并候选(去重): 3个 → LLM 判断
  → LLM: 无重复，新增

去重完成: 新增 6 个, 更新 3 个
```
