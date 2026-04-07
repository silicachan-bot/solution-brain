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
