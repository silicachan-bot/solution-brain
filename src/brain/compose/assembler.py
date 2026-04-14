from __future__ import annotations

from brain.compose.menu import build_menu
from brain.models import PatternCard
from brain.prompts import render_prompt


def assemble_system_prompt(
    soul: str = "",
    patterns: list[PatternCard] | None = None,
) -> str:
    menu = build_menu(patterns or [])
    return render_prompt("compose_system.txt", soul=soul, menu=menu)
