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
