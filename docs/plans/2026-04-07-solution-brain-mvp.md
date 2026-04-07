# solution-brain MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end pipeline that extracts language patterns from Bilibili comments, stores them in a vector database, and assembles prompts with progressive disclosure for chat personality injection.

**Architecture:** Pipeline of 5 modules (`ingest → extract → store → compose → api`), each with clear input/output contracts. Data flows from a SQLite comment database through LLM-based extraction into ChromaDB, then gets retrieved and injected into chat prompts via a tool-use pattern.

**Tech Stack:** Python 3.11, uv, ChromaDB, Qwen3-Embedding-4B (API), OpenAI-compatible LLM API, Jinja2, FastAPI (later)

---

## File Structure

```
solution-brain/
├── pyproject.toml
├── src/brain/
│   ├── __init__.py
│   ├── models.py                 # PatternCard, FrequencyProfile, CleanedComment dataclasses
│   ├── config.py                 # Shared settings (DB paths, API keys, constants)
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── reader.py             # Read comments from bilibili SQLite DB
│   │   ├── cleaner.py            # Filter/deduplicate comments
│   │   └── state.py              # Watermark management (track processing progress)
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── chunker.py            # Split comments into LLM-sized chunks
│   │   └── refiner.py            # LLM extraction + dedup/merge
│   ├── store/
│   │   ├── __init__.py
│   │   ├── pattern_db.py         # ChromaDB persistence for PatternCards
│   │   └── retriever.py          # Semantic search over patterns
│   └── compose/
│       ├── __init__.py
│       ├── menu.py               # Build pattern menu for system prompt
│       ├── tools.py              # inspect_pattern tool definition + handler
│       └── templates/
│           └── system.txt        # Jinja2 system prompt template
├── scripts/
│   └── run_pipeline.py           # CLI entry point for full pipeline
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_ingest.py
│   ├── test_extract.py
│   ├── test_store.py
│   └── test_compose.py
└── data/                         # gitignored runtime data
    └── state.json
```

Each task below produces self-contained, testable changes. Tasks are ordered so each builds on the last.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/brain/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Initialize uv project**

Run:
```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv init --lib --python 3.11 --name brain
```

This creates `pyproject.toml` and `src/brain/__init__.py`.

- [ ] **Step 2: Verify pyproject.toml and adjust**

Read `pyproject.toml`. Then edit it to set the correct metadata and add initial dependencies:

```toml
[project]
name = "brain"
version = "0.1.0"
description = "Language pattern extraction and injection for chat personality"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=1.0.0",
    "openai>=1.0.0",
    "jinja2>=3.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"
```

Note: `openai` is used for both LLM calls and Qwen3 embedding API (OpenAI-compatible endpoints).

- [ ] **Step 3: Update .gitignore**

Append to `.gitignore`:

```
# Runtime data
data/
_storage/
```

- [ ] **Step 4: Create test directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 5: Install dependencies**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv sync
```

Expected: lock file created, dependencies installed.

- [ ] **Step 6: Verify pytest works**

```bash
uv run pytest --version
```

Expected: prints pytest version.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/__init__.py .gitignore .python-version
git commit -m "chore(brain): scaffold uv project with dependencies"
```

---

## Task 2: Data Models

**Files:**
- Create: `src/brain/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for data models**

Create `tests/test_models.py`:

```python
from datetime import datetime
from brain.models import PatternCard, FrequencyProfile, CleanedComment


class TestFrequencyProfile:
    def test_freshness_zero_total(self):
        fp = FrequencyProfile(recent=0, medium=0, long_term=0, total=0)
        assert fp.freshness == 0.0

    def test_freshness_trending_up(self):
        """Recent-heavy pattern should have high freshness."""
        fp = FrequencyProfile(recent=80, medium=120, long_term=150, total=150)
        assert fp.freshness > 0.7

    def test_freshness_dead_pattern(self):
        """Old pattern with no recent activity should have low freshness."""
        fp = FrequencyProfile(recent=0, medium=2, long_term=150, total=300)
        assert fp.freshness < 0.2

    def test_freshness_range(self):
        """Freshness should always be between 0 and 1."""
        for r, m, l, t in [(0, 0, 0, 0), (100, 100, 100, 100), (1, 50, 200, 500)]:
            fp = FrequencyProfile(recent=r, medium=m, long_term=l, total=t)
            assert 0.0 <= fp.freshness <= 1.0


class TestPatternCard:
    def test_create_pattern_card(self):
        fp = FrequencyProfile(recent=10, medium=20, long_term=50, total=80)
        card = PatternCard(
            id="pat-001",
            title="test pattern",
            description="a test",
            template="[A] test",
            examples=["hello test", "world test"],
            frequency=fp,
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        assert card.id == "pat-001"
        assert card.title == "test pattern"
        assert len(card.examples) == 2
        assert card.frequency.freshness > 0

    def test_to_dict_roundtrip(self):
        fp = FrequencyProfile(recent=5, medium=10, long_term=30, total=45)
        card = PatternCard(
            id="pat-002",
            title="roundtrip",
            description="test roundtrip",
            template="[A]...[B]",
            examples=["x...y"],
            frequency=fp,
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        d = card.to_dict()
        restored = PatternCard.from_dict(d)
        assert restored.id == card.id
        assert restored.title == card.title
        assert restored.frequency.recent == card.frequency.recent


class TestCleanedComment:
    def test_create(self):
        c = CleanedComment(
            rpid=123,
            bvid="BV1test",
            uid=456,
            message="hello world",
            ctime=1700000000,
        )
        assert c.rpid == 123
        assert c.message == "hello world"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'brain.models'`

- [ ] **Step 3: Implement models**

Create `src/brain/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class FrequencyProfile:
    recent: int = 0       # last 3 months
    medium: int = 0       # last 6 months
    long_term: int = 0    # last 2 years
    total: int = 0

    @property
    def freshness(self) -> float:
        if self.total == 0:
            return 0.0
        recent_ratio = self.recent / self.total
        medium_ratio = self.medium / self.total
        return recent_ratio * 0.5 + medium_ratio * 0.3 + min(self.total / 500, 1.0) * 0.2


@dataclass
class PatternCard:
    id: str
    title: str
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
            "title": self.title,
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
            title=d["title"],
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


@dataclass
class CleanedComment:
    rpid: int
    bvid: str
    uid: int
    message: str
    ctime: int
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/brain/models.py tests/test_models.py
git commit -m "feat(brain): add PatternCard and FrequencyProfile data models"
```

---

## Task 3: Config Module

**Files:**
- Create: `src/brain/config.py`

- [ ] **Step 1: Create config module**

Create `src/brain/config.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

# Paths
SOLUTION_BRAIN_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SOLUTION_BRAIN_DIR / "data"
STATE_FILE = DATA_DIR / "state.json"
CHROMA_DIR = DATA_DIR / "chromadb"

BILIBILI_DB_PATH = Path(
    os.environ.get(
        "BILIBILI_DB_PATH",
        str(SOLUTION_BRAIN_DIR.parent / "raw-data-pipeline" / "bilibili" / "_storage" / "database" / "bilibili.db"),
    )
)

# LLM API (OpenAI-compatible)
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Embedding API (Qwen3-Embedding-4B, OpenAI-compatible)
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "https://api.deepinfra.com/v1/openai")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBED_DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "2048"))

# Extraction
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "50"))
MIN_COMMENT_LENGTH = int(os.environ.get("MIN_COMMENT_LENGTH", "5"))

# Retrieval
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "8"))
SIMILARITY_WEIGHT = 0.6
FRESHNESS_WEIGHT = 0.4
```

- [ ] **Step 2: Commit**

```bash
git add src/brain/config.py
git commit -m "feat(brain): add config module with paths and API settings"
```

---

## Task 4: Ingest — Reader

**Files:**
- Create: `src/brain/ingest/__init__.py`
- Create: `src/brain/ingest/reader.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests for reader**

Create `tests/test_ingest.py`:

```python
import sqlite3
import tempfile
from pathlib import Path

from brain.models import CleanedComment
from brain.ingest.reader import BilibiliReader


def _create_test_db(path: Path):
    """Create a minimal bilibili-schema SQLite DB for testing."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE videos (
            bvid TEXT PRIMARY KEY,
            title TEXT,
            crawl_status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE comments (
            rpid INTEGER PRIMARY KEY,
            bvid TEXT NOT NULL,
            uid INTEGER,
            uname TEXT,
            message TEXT,
            like INTEGER,
            ctime INTEGER,
            root INTEGER DEFAULT 0,
            parent INTEGER DEFAULT 0,
            rcount INTEGER DEFAULT 0
        )
    """)
    conn.execute("INSERT INTO videos VALUES ('BV1test', 'Test Video', 'completed')")
    conn.execute("INSERT INTO videos VALUES ('BV2test', 'Video 2', 'completed')")
    for i in range(5):
        conn.execute(
            "INSERT INTO comments VALUES (?, 'BV1test', ?, 'user', ?, 10, ?, 0, 0, 0)",
            (i, 100 + i, f"comment {i} from video 1", 1700000000 + i),
        )
    for i in range(3):
        conn.execute(
            "INSERT INTO comments VALUES (?, 'BV2test', ?, 'user', ?, 5, ?, 0, 0, 0)",
            (100 + i, 200 + i, f"comment {i} from video 2", 1700000000 + i),
        )
    conn.commit()
    conn.close()


class TestBilibiliReader:
    def test_list_videos(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)
        reader = BilibiliReader(db_path)
        videos = reader.list_videos()
        assert len(videos) == 2

    def test_read_comments_for_video(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)
        reader = BilibiliReader(db_path)
        comments = reader.read_comments("BV1test")
        assert len(comments) == 5
        assert all(isinstance(c, CleanedComment) for c in comments)
        assert comments[0].bvid == "BV1test"

    def test_read_comments_empty_video(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)
        reader = BilibiliReader(db_path)
        comments = reader.read_comments("BV_nonexistent")
        assert comments == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement reader**

Create `src/brain/ingest/__init__.py`:

```python
```

Create `src/brain/ingest/reader.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

from brain.models import CleanedComment


class BilibiliReader:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def list_videos(self) -> list[dict]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT bvid, title FROM videos WHERE crawl_status = 'completed' ORDER BY bvid"
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def read_comments(self, bvid: str) -> list[CleanedComment]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT rpid, bvid, uid, message, ctime FROM comments WHERE bvid = ? ORDER BY ctime",
                (bvid,),
            )
            return [
                CleanedComment(
                    rpid=row["rpid"],
                    bvid=row["bvid"],
                    uid=row["uid"],
                    message=row["message"],
                    ctime=row["ctime"],
                )
                for row in cur.fetchall()
            ]
        finally:
            conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/brain/ingest/ tests/test_ingest.py
git commit -m "feat(brain): add bilibili comment reader"
```

---

## Task 5: Ingest — Cleaner

**Files:**
- Create: `src/brain/ingest/cleaner.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests for cleaner**

Append to `tests/test_ingest.py`:

```python
from brain.ingest.cleaner import clean_comments
from brain.models import CleanedComment


class TestCleaner:
    def _make_comment(self, rpid: int, message: str, uid: int = 1) -> CleanedComment:
        return CleanedComment(rpid=rpid, bvid="BV1test", uid=uid, message=message, ctime=1700000000)

    def test_removes_short_comments(self):
        comments = [
            self._make_comment(1, "hi"),
            self._make_comment(2, "这是一条足够长的评论"),
        ]
        result = clean_comments(comments)
        assert len(result) == 1
        assert result[0].rpid == 2

    def test_removes_pure_emoji(self):
        comments = [
            self._make_comment(1, "😂😂😂"),
            self._make_comment(2, "这也太好笑了😂"),
        ]
        result = clean_comments(comments)
        assert len(result) == 1
        assert result[0].rpid == 2

    def test_deduplicates_exact_messages(self):
        comments = [
            self._make_comment(1, "这是一条重复的评论", uid=1),
            self._make_comment(2, "这是一条重复的评论", uid=2),
            self._make_comment(3, "这是一条不同的评论", uid=3),
        ]
        result = clean_comments(comments)
        messages = [c.message for c in result]
        assert messages.count("这是一条重复的评论") == 1
        assert "这是一条不同的评论" in messages

    def test_empty_input(self):
        assert clean_comments([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest.py::TestCleaner -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement cleaner**

Create `src/brain/ingest/cleaner.py`:

```python
from __future__ import annotations

import re

from brain.models import CleanedComment
from brain.config import MIN_COMMENT_LENGTH

# Matches strings that are ONLY emoji/symbols (no CJK or Latin text)
_PURE_EMOJI_RE = re.compile(
    r"^[\s\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    r"\U0001f1e0-\U0001f1ff\U00002702-\U000027b0\U0000fe00-\U0000fe0f"
    r"\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\U00002600-\U000026ff"
    r"\U0000200d\U00002b50\U000023f0-\U000023ff\[\]]+$"
)


def _has_text_content(message: str) -> bool:
    """Check if message has actual text (CJK or Latin chars), not just emoji."""
    stripped = message.strip()
    if len(stripped) < MIN_COMMENT_LENGTH:
        return False
    if _PURE_EMOJI_RE.match(stripped):
        return False
    return True


def clean_comments(comments: list[CleanedComment]) -> list[CleanedComment]:
    seen_messages: set[str] = set()
    result: list[CleanedComment] = []

    for comment in comments:
        msg = comment.message.strip()
        if not _has_text_content(msg):
            continue
        if msg in seen_messages:
            continue
        seen_messages.add(msg)
        result.append(comment)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: all 7 tests PASS (3 reader + 4 cleaner).

- [ ] **Step 5: Commit**

```bash
git add src/brain/ingest/cleaner.py tests/test_ingest.py
git commit -m "feat(brain): add comment cleaner with dedup and filtering"
```

---

## Task 6: Ingest — Watermark State

**Files:**
- Create: `src/brain/ingest/state.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests for state**

Append to `tests/test_ingest.py`:

```python
from brain.ingest.state import WatermarkState


class TestWatermarkState:
    def test_get_default_watermark(self, tmp_path):
        state = WatermarkState(tmp_path / "state.json")
        assert state.get_watermark("bilibili") is None

    def test_set_and_get_watermark(self, tmp_path):
        state = WatermarkState(tmp_path / "state.json")
        state.set_watermark("bilibili", "BV1test")
        assert state.get_watermark("bilibili") == "BV1test"

    def test_persistence(self, tmp_path):
        path = tmp_path / "state.json"
        state1 = WatermarkState(path)
        state1.set_watermark("bilibili", "BV1abc")

        state2 = WatermarkState(path)
        assert state2.get_watermark("bilibili") == "BV1abc"

    def test_multiple_sources(self, tmp_path):
        state = WatermarkState(tmp_path / "state.json")
        state.set_watermark("bilibili", "BV1test")
        state.set_watermark("other_source", "id-999")
        assert state.get_watermark("bilibili") == "BV1test"
        assert state.get_watermark("other_source") == "id-999"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest.py::TestWatermarkState -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement state**

Create `src/brain/ingest/state.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


class WatermarkState:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._data: dict = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def get_watermark(self, source: str) -> str | None:
        return self._data.get(source)

    def set_watermark(self, source: str, value: str) -> None:
        self._data[source] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/brain/ingest/state.py tests/test_ingest.py
git commit -m "feat(brain): add watermark state for incremental processing"
```

---

## Task 7: Extract — Chunker

**Files:**
- Create: `src/brain/extract/__init__.py`
- Create: `src/brain/extract/chunker.py`
- Create: `tests/test_extract.py`

- [ ] **Step 1: Write failing tests for chunker**

Create `tests/test_extract.py`:

```python
from brain.models import CleanedComment
from brain.extract.chunker import chunk_comments


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_extract.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement chunker**

Create `src/brain/extract/__init__.py`:

```python
```

Create `src/brain/extract/chunker.py`:

```python
from __future__ import annotations

from brain.models import CleanedComment


def chunk_comments(
    comments: list[CleanedComment], chunk_size: int = 50
) -> list[list[str]]:
    if not comments:
        return []

    messages = [c.message for c in comments]
    return [
        messages[i : i + chunk_size]
        for i in range(0, len(messages), chunk_size)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_extract.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/brain/extract/ tests/test_extract.py
git commit -m "feat(brain): add comment chunker for LLM-sized batches"
```

---

## Task 8: Extract — LLM Refiner

**Files:**
- Create: `src/brain/extract/refiner.py`
- Modify: `tests/test_extract.py`

- [ ] **Step 1: Write failing tests for refiner**

The refiner makes LLM API calls, so tests use a mock. Append to `tests/test_extract.py`:

```python
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

from brain.extract.refiner import extract_from_chunk, deduplicate_and_merge
from brain.models import PatternCard, FrequencyProfile


class TestExtractFromChunk:
    def test_parses_llm_response(self):
        mock_patterns = [
            {
                "title": "好家伙式吐槽",
                "template": "[A]...好家伙...",
                "examples": ["这也行...好家伙...", "又来了...好家伙..."],
                "description": "表达无奈和吐槽",
            }
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_patterns, ensure_ascii=False)

        with patch("brain.extract.refiner._call_llm", return_value=mock_response):
            results = extract_from_chunk(["comment1", "comment2"])

        assert len(results) == 1
        assert results[0].title == "好家伙式吐槽"
        assert results[0].template == "[A]...好家伙..."

    def test_handles_empty_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[]"

        with patch("brain.extract.refiner._call_llm", return_value=mock_response):
            results = extract_from_chunk(["comment1", "comment2"])

        assert results == []


class TestDeduplicateAndMerge:
    def _make_card(self, id: str, title: str, recent: int = 5) -> PatternCard:
        return PatternCard(
            id=id,
            title=title,
            description=f"desc for {title}",
            template=f"[A] {title}",
            examples=[f"example of {title}"],
            frequency=FrequencyProfile(recent=recent, medium=recent, long_term=recent, total=recent),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_no_duplicates(self):
        cards = [self._make_card("1", "pattern A"), self._make_card("2", "pattern B")]
        new, updates = deduplicate_and_merge(cards, existing=[])
        assert len(new) == 2
        assert len(updates) == 0

    def test_merges_similar_new_cards(self):
        cards = [
            self._make_card("1", "好家伙式吐槽", recent=3),
            self._make_card("2", "好家伙式吐槽", recent=5),
        ]
        new, updates = deduplicate_and_merge(cards, existing=[])
        # Two identical titles should merge into one
        assert len(new) == 1
        assert new[0].frequency.recent == 8  # summed

    def test_updates_existing_pattern(self):
        existing = [self._make_card("existing-1", "好家伙式吐槽", recent=10)]
        new_cards = [self._make_card("new-1", "好家伙式吐槽", recent=3)]
        new, updates = deduplicate_and_merge(new_cards, existing=existing)
        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].id == "existing-1"
        assert updates[0].frequency.recent == 13
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_extract.py::TestExtractFromChunk -v
uv run pytest tests/test_extract.py::TestDeduplicateAndMerge -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement refiner**

Create `src/brain/extract/refiner.py`:

```python
from __future__ import annotations

import json
import uuid
from datetime import datetime

from openai import OpenAI

from brain.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from brain.models import PatternCard, FrequencyProfile

_EXTRACT_PROMPT = """\
以下是 B 站某视频下的用户评论。请从中发现值得收录的语言模式——特别是：
- 多人使用的相似句式
- 不像 AI 会自然生成的表达
- 可以替换内容复用的句式模板

对每个发现的模式，输出 JSON 数组，每个元素包含：
{
  "title": "简短标题",
  "template": "含 [A] [B] 占位符的模板句",
  "examples": ["2-5个真实例句"],
  "description": "模式描述：是什么、什么时候用、传达什么感觉"
}

如果这批评论中没有值得收录的模式，返回空数组 []。
只输出 JSON，不要其他文字。

评论：
"""


def _call_llm(prompt: str):
    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
    return client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )


def extract_from_chunk(messages: list[str]) -> list[PatternCard]:
    numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))
    prompt = _EXTRACT_PROMPT + numbered

    response = _call_llm(prompt)
    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        raw_patterns = json.loads(content)
    except json.JSONDecodeError:
        return []

    if not isinstance(raw_patterns, list):
        return []

    now = datetime.now()
    cards = []
    for p in raw_patterns:
        if not all(k in p for k in ("title", "template", "examples", "description")):
            continue
        cards.append(
            PatternCard(
                id=f"pat-{uuid.uuid4().hex[:8]}",
                title=p["title"],
                description=p["description"],
                template=p["template"],
                examples=p["examples"][:5],
                frequency=FrequencyProfile(recent=1, medium=1, long_term=1, total=1),
                source="bilibili",
                created_at=now,
                updated_at=now,
            )
        )
    return cards


def deduplicate_and_merge(
    cards: list[PatternCard],
    existing: list[PatternCard],
) -> tuple[list[PatternCard], list[PatternCard]]:
    """Deduplicate new cards among themselves and against existing patterns.

    Returns (new_cards, updated_existing_cards).

    Uses title matching for dedup. A future improvement could use
    embedding similarity, but title matching is sufficient for MVP.
    """
    # Index existing by normalized title
    existing_by_title: dict[str, PatternCard] = {}
    for card in existing:
        existing_by_title[card.title.strip().lower()] = card

    # Merge new cards among themselves first
    merged: dict[str, PatternCard] = {}
    for card in cards:
        key = card.title.strip().lower()
        if key in merged:
            _merge_into(merged[key], card)
        else:
            merged[key] = card

    # Split into new vs updates-to-existing
    new_cards: list[PatternCard] = []
    updated: list[PatternCard] = []

    for key, card in merged.items():
        if key in existing_by_title:
            target = existing_by_title[key]
            _merge_into(target, card)
            updated.append(target)
        else:
            new_cards.append(card)

    return new_cards, updated


def _merge_into(target: PatternCard, source: PatternCard) -> None:
    """Merge source pattern data into target (mutates target)."""
    target.frequency.recent += source.frequency.recent
    target.frequency.medium += source.frequency.medium
    target.frequency.long_term += source.frequency.long_term
    target.frequency.total += source.frequency.total

    existing_examples = set(target.examples)
    for ex in source.examples:
        if ex not in existing_examples and len(target.examples) < 5:
            target.examples.append(ex)

    target.updated_at = datetime.now()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_extract.py -v
```

Expected: all 7 tests PASS (4 chunker + 3 refiner).

- [ ] **Step 5: Commit**

```bash
git add src/brain/extract/refiner.py tests/test_extract.py
git commit -m "feat(brain): add LLM-based pattern extraction and dedup"
```

---

## Task 9: Store — PatternDB (ChromaDB)

**Files:**
- Create: `src/brain/store/__init__.py`
- Create: `src/brain/store/pattern_db.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing tests for pattern_db**

Create `tests/test_store.py`:

```python
from datetime import datetime

from brain.models import PatternCard, FrequencyProfile
from brain.store.pattern_db import PatternDB


def _make_card(id: str, title: str, description: str = "test desc") -> PatternCard:
    return PatternCard(
        id=id,
        title=title,
        description=description,
        template=f"[A] {title}",
        examples=[f"example of {title}"],
        frequency=FrequencyProfile(recent=5, medium=10, long_term=30, total=45),
        source="test",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


class TestPatternDB:
    def test_save_and_get(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        card = _make_card("pat-001", "test pattern")
        db.save([card])

        retrieved = db.get("pat-001")
        assert retrieved is not None
        assert retrieved.title == "test pattern"
        assert retrieved.frequency.recent == 5

    def test_list_all(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        db.save([_make_card("1", "alpha"), _make_card("2", "beta")])
        all_cards = db.list_all()
        assert len(all_cards) == 2

    def test_update(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        card = _make_card("pat-001", "test pattern")
        db.save([card])

        card.frequency.recent = 99
        card.examples.append("new example")
        db.update([card])

        retrieved = db.get("pat-001")
        assert retrieved.frequency.recent == 99
        assert "new example" in retrieved.examples

    def test_save_empty_list(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        db.save([])
        assert db.list_all() == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_store.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement pattern_db**

Create `src/brain/store/__init__.py`:

```python
```

Create `src/brain/store/pattern_db.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import chromadb

from brain.models import PatternCard


class PatternDB:
    """Stores PatternCards with ChromaDB for vector search and JSON metadata for full data."""

    COLLECTION_NAME = "patterns"

    def __init__(self, persist_dir: Path | str):
        self.persist_dir = Path(persist_dir)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, cards: list[PatternCard]) -> None:
        if not cards:
            return
        self._collection.upsert(
            ids=[c.id for c in cards],
            documents=[self._embed_text(c) for c in cards],
            metadatas=[{"json": json.dumps(c.to_dict(), ensure_ascii=False)} for c in cards],
        )

    def update(self, cards: list[PatternCard]) -> None:
        self.save(cards)

    def get(self, pattern_id: str) -> PatternCard | None:
        try:
            result = self._collection.get(ids=[pattern_id], include=["metadatas"])
        except Exception:
            return None
        if not result["ids"]:
            return None
        raw = json.loads(result["metadatas"][0]["json"])
        return PatternCard.from_dict(raw)

    def list_all(self) -> list[PatternCard]:
        result = self._collection.get(include=["metadatas"])
        if not result["ids"]:
            return []
        return [
            PatternCard.from_dict(json.loads(m["json"]))
            for m in result["metadatas"]
        ]

    def query(self, query_text: str, top_k: int = 16) -> list[tuple[PatternCard, float]]:
        """Query by text using ChromaDB's built-in embedding.

        Returns list of (PatternCard, distance) tuples.
        """
        count = self._collection.count()
        if count == 0:
            return []
        n = min(top_k, count)
        result = self._collection.query(
            query_texts=[query_text],
            n_results=n,
            include=["metadatas", "distances"],
        )
        cards = []
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            card = PatternCard.from_dict(json.loads(meta["json"]))
            similarity = 1.0 - dist  # cosine distance → similarity
            cards.append((card, similarity))
        return cards

    @staticmethod
    def _embed_text(card: PatternCard) -> str:
        """Text used for embedding: description + examples."""
        examples_text = " / ".join(card.examples)
        return f"{card.description} 例句：{examples_text}"
```

Note: For MVP, this uses ChromaDB's built-in default embedding. Task 11 will switch to Qwen3-Embedding-4B.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_store.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/brain/store/ tests/test_store.py
git commit -m "feat(brain): add ChromaDB-backed pattern storage"
```

---

## Task 10: Store — Retriever

**Files:**
- Create: `src/brain/store/retriever.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: Write failing tests for retriever**

Append to `tests/test_store.py`:

```python
from brain.store.retriever import retrieve_patterns


class TestRetriever:
    def test_retrieve_returns_cards(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        db.save([
            _make_card("1", "吐槽表达", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "撒娇语气", "用叠字和语气词表达撒娇"),
            _make_card("3", "反问句式", "用反问表达不满"),
        ])
        results = retrieve_patterns(db, "这也太离谱了吧", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(c, PatternCard) for c in results)

    def test_retrieve_empty_db(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        results = retrieve_patterns(db, "test query", top_k=5)
        assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_store.py::TestRetriever -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement retriever**

Create `src/brain/store/retriever.py`:

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
    raw_results = db.query(conversation_text, top_k=top_k * 2)
    if not raw_results:
        return []

    scored = [
        (card, similarity * SIMILARITY_WEIGHT + card.frequency.freshness * FRESHNESS_WEIGHT)
        for card, similarity in raw_results
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [card for card, _ in scored[:top_k]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_store.py -v
```

Expected: all 6 tests PASS (4 db + 2 retriever).

- [ ] **Step 5: Commit**

```bash
git add src/brain/store/retriever.py tests/test_store.py
git commit -m "feat(brain): add pattern retriever with freshness weighting"
```

---

## Task 11: Qwen3 Embedding Integration

**Files:**
- Create: `src/brain/store/embedding.py`
- Modify: `src/brain/store/pattern_db.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: Write failing test for embedding function**

Append to `tests/test_store.py`:

```python
from unittest.mock import patch, MagicMock
from brain.store.embedding import QwenEmbeddingFunction


class TestQwenEmbedding:
    def test_call_returns_embeddings(self):
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 2048),
            MagicMock(embedding=[0.2] * 2048),
        ]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("brain.store.embedding.OpenAI", return_value=mock_client):
            ef = QwenEmbeddingFunction()
            result = ef(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 2048
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_store.py::TestQwenEmbedding -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement embedding function**

Create `src/brain/store/embedding.py`:

```python
from __future__ import annotations

from chromadb import EmbeddingFunction, Documents, Embeddings
from openai import OpenAI

from brain.config import EMBED_API_BASE, EMBED_API_KEY, EMBED_MODEL, EMBED_DIMENSIONS


class QwenEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self._client = OpenAI(base_url=EMBED_API_BASE, api_key=EMBED_API_KEY)

    def __call__(self, input: Documents) -> Embeddings:
        response = self._client.embeddings.create(
            model=EMBED_MODEL,
            input=input,
            dimensions=EMBED_DIMENSIONS,
        )
        return [item.embedding for item in response.data]
```

- [ ] **Step 4: Update PatternDB to use Qwen embedding**

Modify `src/brain/store/pattern_db.py`. Replace the `__init__` method:

```python
from brain.store.embedding import QwenEmbeddingFunction
```

And change `__init__`:

```python
    def __init__(self, persist_dir: Path | str, embedding_fn=None):
        self.persist_dir = Path(persist_dir)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._embedding_fn = embedding_fn
        kwargs = {
            "name": self.COLLECTION_NAME,
            "metadata": {"hnsw:space": "cosine"},
        }
        if self._embedding_fn is not None:
            kwargs["embedding_function"] = self._embedding_fn
        self._collection = self._client.get_or_create_collection(**kwargs)
```

This keeps existing tests working (they pass `embedding_fn=None` and use ChromaDB's default), while production code can inject the Qwen function.

- [ ] **Step 5: Run all store tests**

```bash
uv run pytest tests/test_store.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/brain/store/embedding.py src/brain/store/pattern_db.py tests/test_store.py
git commit -m "feat(brain): add Qwen3-Embedding-4B integration for ChromaDB"
```

---

## Task 12: Compose — Menu Builder

**Files:**
- Create: `src/brain/compose/__init__.py`
- Create: `src/brain/compose/menu.py`
- Create: `tests/test_compose.py`

- [ ] **Step 1: Write failing tests for menu builder**

Create `tests/test_compose.py`:

```python
from datetime import datetime

from brain.models import PatternCard, FrequencyProfile
from brain.compose.menu import build_menu


def _make_card(id: str, title: str, template: str) -> PatternCard:
    return PatternCard(
        id=id,
        title=title,
        description=f"desc for {title}",
        template=template,
        examples=[f"example of {title}"],
        frequency=FrequencyProfile(recent=5, medium=10, long_term=30, total=45),
        source="test",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


class TestBuildMenu:
    def test_formats_pattern_list(self):
        cards = [
            _make_card("pat-001", "好家伙式吐槽", "[A]...好家伙..."),
            _make_card("pat-002", "叠字撒娇", "[A]嘛~[B]啦~"),
        ]
        menu = build_menu(cards)
        assert "[pat-001]" in menu
        assert "好家伙式吐槽" in menu
        assert "[A]...好家伙..." in menu
        assert "[pat-002]" in menu

    def test_empty_patterns(self):
        menu = build_menu([])
        assert "没有" in menu or "无" in menu or menu.strip() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_compose.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement menu builder**

Create `src/brain/compose/__init__.py`:

```python
```

Create `src/brain/compose/menu.py`:

```python
from __future__ import annotations

from brain.models import PatternCard


def build_menu(patterns: list[PatternCard]) -> str:
    if not patterns:
        return "当前没有可用的语言模式。"

    lines = []
    for i, p in enumerate(patterns, 1):
        lines.append(f'{i}. [{p.id}] {p.title} — "{p.template}"')
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_compose.py -v
```

Expected: all 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/brain/compose/ tests/test_compose.py
git commit -m "feat(brain): add pattern menu builder for system prompt"
```

---

## Task 13: Compose — Tool Definition and Handler

**Files:**
- Create: `src/brain/compose/tools.py`
- Modify: `tests/test_compose.py`

- [ ] **Step 1: Write failing tests for tool handler**

Append to `tests/test_compose.py`:

```python
from brain.compose.tools import get_tool_definition, handle_inspect_pattern
from brain.store.pattern_db import PatternDB


class TestToolDefinition:
    def test_tool_schema(self):
        tool = get_tool_definition()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "inspect_pattern"
        assert "pattern_id" in tool["function"]["parameters"]["properties"]


class TestHandleInspectPattern:
    def test_returns_pattern_details(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        card = _make_card("pat-001", "好家伙式吐槽", "[A]...好家伙...")
        db.save([card])

        result = handle_inspect_pattern(db, "pat-001")
        assert "好家伙式吐槽" in result
        assert "[A]...好家伙..." in result
        assert "desc for" in result

    def test_pattern_not_found(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        result = handle_inspect_pattern(db, "nonexistent")
        assert "找不到" in result or "not found" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_compose.py::TestToolDefinition -v
uv run pytest tests/test_compose.py::TestHandleInspectPattern -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tools**

Create `src/brain/compose/tools.py`:

```python
from __future__ import annotations

from brain.store.pattern_db import PatternDB


def get_tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "inspect_pattern",
            "description": "查看某个语言模式的详细描述和使用示例。使用任何模式前必须先查看。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_id": {
                        "type": "string",
                        "description": "模式 ID，如 pat-a3f2c1",
                    }
                },
                "required": ["pattern_id"],
            },
        },
    }


def handle_inspect_pattern(db: PatternDB, pattern_id: str) -> str:
    card = db.get(pattern_id)
    if card is None:
        return f"找不到 ID 为 {pattern_id} 的语言模式。"

    examples_text = "\n".join(f"  - {ex}" for ex in card.examples)
    return (
        f"【{card.title}】\n"
        f"描述：{card.description}\n"
        f"模板：{card.template}\n"
        f"例句：\n{examples_text}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_compose.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/brain/compose/tools.py tests/test_compose.py
git commit -m "feat(brain): add inspect_pattern tool definition and handler"
```

---

## Task 14: Compose — System Prompt Template

**Files:**
- Create: `src/brain/compose/templates/system.txt`
- Create: `src/brain/compose/assembler.py`
- Modify: `tests/test_compose.py`

- [ ] **Step 1: Write failing tests for assembler**

Append to `tests/test_compose.py`:

```python
from brain.compose.assembler import assemble_system_prompt


class TestAssembler:
    def test_assembles_with_patterns(self):
        cards = [
            _make_card("pat-001", "好家伙式吐槽", "[A]...好家伙..."),
            _make_card("pat-002", "叠字撒娇", "[A]嘛~[B]啦~"),
        ]
        prompt = assemble_system_prompt(cards)
        assert "好家伙式吐槽" in prompt
        assert "叠字撒娇" in prompt
        assert "inspect_pattern" in prompt or "查看" in prompt

    def test_assembles_without_patterns(self):
        prompt = assemble_system_prompt([])
        assert isinstance(prompt, str)
        assert len(prompt) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_compose.py::TestAssembler -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create system prompt template**

Create `src/brain/compose/templates/system.txt`:

```
你有以下语言模式可以使用。回复前你可以调用 inspect_pattern 工具查看感兴趣的模式了解详情，然后决定是否使用。你也完全可以不使用任何模式，用你自然的方式回复。

可用模式：
{{ menu }}
```

- [ ] **Step 4: Implement assembler**

Create `src/brain/compose/assembler.py`:

```python
from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from brain.models import PatternCard
from brain.compose.menu import build_menu

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "system.txt"


def assemble_system_prompt(patterns: list[PatternCard]) -> str:
    menu = build_menu(patterns)
    template_text = _TEMPLATE_PATH.read_text()
    template = Template(template_text)
    return template.render(menu=menu)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_compose.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/brain/compose/assembler.py src/brain/compose/templates/ tests/test_compose.py
git commit -m "feat(brain): add system prompt assembler with Jinja2 template"
```

---

## Task 15: Pipeline Script

**Files:**
- Create: `scripts/run_pipeline.py`

- [ ] **Step 1: Implement pipeline script**

Create `scripts/run_pipeline.py`:

```python
"""
Full extraction pipeline: ingest → extract → store.

Usage:
    uv run python scripts/run_pipeline.py                    # incremental
    uv run python scripts/run_pipeline.py --full             # full reprocess
    uv run python scripts/run_pipeline.py --limit 5          # limit to N videos
    uv run python scripts/run_pipeline.py --chunk-size 30    # comments per chunk
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path so `brain` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain.config import BILIBILI_DB_PATH, DATA_DIR, CHROMA_DIR, CHUNK_SIZE, STATE_FILE
from brain.ingest.reader import BilibiliReader
from brain.ingest.cleaner import clean_comments
from brain.ingest.state import WatermarkState
from brain.extract.chunker import chunk_comments
from brain.extract.refiner import extract_from_chunk, deduplicate_and_merge
from brain.store.pattern_db import PatternDB
from brain.store.embedding import QwenEmbeddingFunction


def main():
    parser = argparse.ArgumentParser(description="Run the pattern extraction pipeline")
    parser.add_argument("--full", action="store_true", help="Full reprocess (ignore watermark)")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0=all)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Comments per chunk")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing LLM calls")
    args = parser.parse_args()

    # Init
    reader = BilibiliReader(BILIBILI_DB_PATH)
    state = WatermarkState(STATE_FILE)
    db = PatternDB(CHROMA_DIR, embedding_fn=QwenEmbeddingFunction())

    # 1. List videos
    videos = reader.list_videos()
    print(f"Found {len(videos)} completed videos in database")

    # 2. Filter by watermark (incremental)
    watermark = None if args.full else state.get_watermark("bilibili")
    if watermark:
        videos = [v for v in videos if v["bvid"] > watermark]
        print(f"After watermark filter: {len(videos)} new videos")

    if args.limit > 0:
        videos = videos[: args.limit]
        print(f"Limited to {len(videos)} videos")

    if not videos:
        print("Nothing to process.")
        return

    # 3. Process each video
    all_patterns = []
    for i, video in enumerate(videos, 1):
        bvid = video["bvid"]
        title = video.get("title", "untitled")
        print(f"\n[{i}/{len(videos)}] {bvid} — {title}")

        comments = reader.read_comments(bvid)
        print(f"  Raw comments: {len(comments)}")

        cleaned = clean_comments(comments)
        print(f"  After cleaning: {len(cleaned)}")

        if not cleaned:
            continue

        chunks = chunk_comments(cleaned, chunk_size=args.chunk_size)
        print(f"  Chunks: {len(chunks)}")

        if args.dry_run:
            print("  [dry-run] Skipping LLM extraction")
            continue

        for j, chunk in enumerate(chunks, 1):
            print(f"  Extracting chunk {j}/{len(chunks)}...", end=" ", flush=True)
            patterns = extract_from_chunk(chunk)
            print(f"found {len(patterns)} patterns")
            all_patterns.extend(patterns)

    if args.dry_run:
        print(f"\n[dry-run] Would process {len(all_patterns)} raw patterns total")
        return

    # 4. Deduplicate and merge
    print(f"\nRaw patterns extracted: {len(all_patterns)}")
    existing = db.list_all()
    print(f"Existing patterns in DB: {len(existing)}")

    new_cards, updates = deduplicate_and_merge(all_patterns, existing)
    print(f"New patterns: {len(new_cards)}, Updated: {len(updates)}")

    db.save(new_cards)
    db.update(updates)

    # 5. Update watermark
    if videos:
        last_bvid = videos[-1]["bvid"]
        state.set_watermark("bilibili", last_bvid)
        print(f"Watermark updated to: {last_bvid}")

    total = len(existing) + len(new_cards)
    print(f"\nDone. Total patterns in DB: {total}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test dry-run mode against real database**

```bash
cd /home/sparidae/projects/silicachan/solution-brain
uv run python scripts/run_pipeline.py --dry-run --limit 2
```

Expected: prints video list, comment counts, chunk counts — no LLM calls made. Verifies the ingest pipeline works end-to-end with real data.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat(brain): add pipeline CLI with incremental processing"
```

---

## Task 16: End-to-End Smoke Test

**Files:**
- Modify: `tests/test_extract.py` (add integration test, skipped by default)

- [ ] **Step 1: Add integration test (skip unless API key set)**

Append to `tests/test_extract.py`:

```python
import os
import pytest

from brain.extract.refiner import extract_from_chunk


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
        results = extract_from_chunk(comments)
        assert isinstance(results, list)
        # Should find at least the "好家伙" pattern
        if results:
            card = results[0]
            assert card.title
            assert card.template
            assert len(card.examples) > 0
```

- [ ] **Step 2: Run unit tests (no API key needed)**

```bash
uv run pytest tests/ -v -k "not Integration"
```

Expected: all unit tests PASS.

- [ ] **Step 3: Run integration test (requires API key)**

```bash
LLM_API_KEY=<your-key> LLM_API_BASE=<your-base-url> uv run pytest tests/test_extract.py::TestExtractIntegration -v
```

Expected: PASS — real LLM returns structured patterns.

- [ ] **Step 4: Commit**

```bash
git add tests/test_extract.py
git commit -m "test(brain): add LLM extraction integration smoke test"
```

---

## Task 17: Create .env.example

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create example env file**

Create `.env.example`:

```bash
# LLM API (for pattern extraction)
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=gpt-4o-mini

# Embedding API (Qwen3-Embedding-4B, OpenAI-compatible endpoint)
EMBED_API_BASE=https://api.deepinfra.com/v1/openai
EMBED_API_KEY=xxx
EMBED_MODEL=Qwen/Qwen3-Embedding-4B
EMBED_DIMENSIONS=2048

# Optional overrides
# BILIBILI_DB_PATH=/path/to/bilibili.db
# CHUNK_SIZE=50
# MIN_COMMENT_LENGTH=5
# RETRIEVAL_TOP_K=8
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "doc(brain): add .env.example with API configuration"
```
