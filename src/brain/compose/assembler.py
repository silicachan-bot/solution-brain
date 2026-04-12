from __future__ import annotations

from brain.compose.menu import build_menu
from brain.models import PatternCard
from brain.prompts import render_prompt


def assemble_system_prompt(patterns: list[PatternCard]) -> str:
    menu = build_menu(patterns)
    return render_prompt("compose_system.txt", menu=menu)
