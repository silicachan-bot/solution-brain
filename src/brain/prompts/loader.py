from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from jinja2 import Template


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return files("brain.prompts").joinpath(name).read_text(encoding="utf-8")


def render_prompt(name: str, **context: object) -> str:
    template = Template(load_prompt(name))
    return template.render(**context)
