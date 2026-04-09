from __future__ import annotations

from brain.models import PatternCard


def build_menu(patterns: list[PatternCard]) -> str:
    if not patterns:
        return "当前没有可用的语言模式。"

    lines = []
    for i, p in enumerate(patterns, 1):
        lines.append(f'{i}. [{p.id}] "{p.template}" — {p.description[:30]}')
    return "\n".join(lines)
