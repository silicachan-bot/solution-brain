from __future__ import annotations

import json
from pathlib import Path

import chromadb

from brain.models import PatternCard


class PatternDB:
    COLLECTION_NAME = "patterns"

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
            similarity = 1.0 - dist
            cards.append((card, similarity))
        return cards

    @staticmethod
    def _embed_text(card: PatternCard) -> str:
        examples_text = " / ".join(card.examples)
        return f"{card.description} 例句：{examples_text}"
