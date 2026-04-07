from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from brain.models import PatternCard
from brain.compose.menu import build_menu

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "system.txt"


def assemble_system_prompt(patterns: list[PatternCard]) -> str:
    menu = build_menu(patterns)
    template_text = _TEMPLATE_PATH.read_text()
    template = Template(template_text)
    return template.render(menu=menu)
