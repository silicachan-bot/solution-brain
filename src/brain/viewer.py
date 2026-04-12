from __future__ import annotations

from brain.models import PatternCard, PatternOrigin


_SORT_OPTIONS = {"updated_at", "freshness", "template"}


def filter_patterns(patterns: list[PatternCard], query: str) -> list[PatternCard]:
    needle = query.strip().lower()
    if not needle:
        return list(patterns)

    result = []
    for card in patterns:
        haystack = "\n".join(
            [
                card.template,
                card.description,
                *card.examples,
                *[
                    "\n".join([
                        origin.example,
                        origin.bvid,
                        origin.video_title,
                        origin.parent_message,
                        origin.reply_message,
                    ])
                    for origin in card.origins
                ],
            ]
        ).lower()
        if needle in haystack:
            result.append(card)
    return result



def sort_patterns(patterns: list[PatternCard], sort_by: str) -> list[PatternCard]:
    if sort_by not in _SORT_OPTIONS:
        sort_by = "updated_at"

    if sort_by == "template":
        return sorted(patterns, key=lambda card: card.template)
    if sort_by == "freshness":
        return sorted(
            patterns,
            key=lambda card: card.frequency.freshness,
            reverse=True,
        )
    return sorted(patterns, key=lambda card: card.updated_at, reverse=True)



def format_pattern_summary(card: PatternCard) -> str:
    return (
        f"{card.template} | "
        f"freshness={card.frequency.freshness:.2f} | total={card.frequency.total}"
    )


def group_origins_by_example(card: PatternCard) -> dict[str, list[PatternOrigin]]:
    grouped = {example: [] for example in card.examples}
    for origin in card.origins:
        if origin.example in grouped:
            grouped[origin.example].append(origin)
    return grouped
