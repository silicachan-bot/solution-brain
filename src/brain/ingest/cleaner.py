from __future__ import annotations

import re

from brain.models import CleanedComment
from brain.config import MIN_COMMENT_LENGTH

_PURE_EMOJI_RE = re.compile(
    r"^[\s\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    r"\U0001f1e0-\U0001f1ff\U00002702-\U000027b0\U0000fe00-\U0000fe0f"
    r"\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\U00002600-\U000026ff"
    r"\U0000200d\U00002b50\U000023f0-\U000023ff\[\]]+$"
)


def _has_text_content(message: str) -> bool:
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
