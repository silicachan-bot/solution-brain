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
        try:
            self._table = self._db.open_table(self.TABLE_NAME)
        except Exception:
            self._table = None

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
