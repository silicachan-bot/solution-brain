from __future__ import annotations

from brain.models import CleanedComment, CommentPair


def build_comment_pairs(comments: list[CleanedComment]) -> list[CommentPair]:
    if not comments:
        return []

    by_rpid = {comment.rpid: comment for comment in comments}
    pairs: list[CommentPair] = []
    seen: set[tuple[int, int]] = set()

    for comment in comments:
        if comment.root == 0 and comment.parent == 0:
            continue

        anchor_rpid = comment.parent or comment.root
        parent = by_rpid.get(anchor_rpid)
        if parent is None or parent.rpid == comment.rpid:
            continue

        key = (parent.rpid, comment.rpid)
        if key in seen:
            continue

        seen.add(key)
        pairs.append(CommentPair(parent=parent, reply=comment))

    return pairs


def format_comment_pair(pair: CommentPair) -> str:
    return (
        f"上文评论：{pair.parent.message}\n"
        f"回复评论：{pair.reply.message}"
    )


def chunk_comments(
    comment_pairs: list[CommentPair], chunk_size: int = 50
) -> list[list[str]]:
    if not comment_pairs:
        return []

    messages = [format_comment_pair(pair) for pair in comment_pairs]
    return [
        messages[i : i + chunk_size]
        for i in range(0, len(messages), chunk_size)
    ]
