from __future__ import annotations

import json
import uuid
from datetime import datetime

from openai import OpenAI

from brain.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from brain.models import PatternCard, FrequencyProfile

_EXTRACT_PROMPT = """\
以下是 B 站某视频下的用户评论。请从中发现值得收录的语言模式——特别是：
- 多人使用的相似句式
- 不像 AI 会自然生成的表达
- 可以替换内容复用的句式模板

对每个发现的模式，输出 JSON 数组，每个元素包含：
{
  "title": "简短标题",
  "template": "含 [A] [B] 占位符的模板句",
  "examples": ["2-5个真实例句"],
  "description": "模式描述：是什么、什么时候用、传达什么感觉"
}

如果这批评论中没有值得收录的模式，返回空数组 []。
只输出 JSON，不要其他文字。

评论：
"""


def _call_llm(prompt: str):
    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
    return client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )


def extract_from_chunk(messages: list[str]) -> list[PatternCard]:
    numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))
    prompt = _EXTRACT_PROMPT + numbered

    response = _call_llm(prompt)
    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        raw_patterns = json.loads(content)
    except json.JSONDecodeError:
        return []

    if not isinstance(raw_patterns, list):
        return []

    now = datetime.now()
    cards = []
    for p in raw_patterns:
        if not all(k in p for k in ("title", "template", "examples", "description")):
            continue
        cards.append(
            PatternCard(
                id=f"pat-{uuid.uuid4().hex[:8]}",
                title=p["title"],
                description=p["description"],
                template=p["template"],
                examples=p["examples"][:5],
                frequency=FrequencyProfile(recent=1, medium=1, long_term=1, total=1),
                source="bilibili",
                created_at=now,
                updated_at=now,
            )
        )
    return cards


def deduplicate_and_merge(
    cards: list[PatternCard],
    existing: list[PatternCard],
) -> tuple[list[PatternCard], list[PatternCard]]:
    """Deduplicate new cards among themselves and against existing patterns.

    Returns (new_cards, updated_existing_cards).
    Uses title matching for dedup (MVP; could use embedding similarity later).
    """
    existing_by_title: dict[str, PatternCard] = {}
    for card in existing:
        existing_by_title[card.title.strip().lower()] = card

    merged: dict[str, PatternCard] = {}
    for card in cards:
        key = card.title.strip().lower()
        if key in merged:
            _merge_into(merged[key], card)
        else:
            merged[key] = card

    new_cards: list[PatternCard] = []
    updated: list[PatternCard] = []

    for key, card in merged.items():
        if key in existing_by_title:
            target = existing_by_title[key]
            _merge_into(target, card)
            updated.append(target)
        else:
            new_cards.append(card)

    return new_cards, updated


def _merge_into(target: PatternCard, source: PatternCard) -> None:
    target.frequency.recent += source.frequency.recent
    target.frequency.medium += source.frequency.medium
    target.frequency.long_term += source.frequency.long_term
    target.frequency.total += source.frequency.total

    existing_examples = set(target.examples)
    for ex in source.examples:
        if ex not in existing_examples and len(target.examples) < 5:
            target.examples.append(ex)

    target.updated_at = datetime.now()
