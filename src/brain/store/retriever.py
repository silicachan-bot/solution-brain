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
