from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from openai import OpenAI

from brain.config import DATA_DIR, LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from brain.models import PatternCard, FrequencyProfile

# File logger for full LLM responses (background log, not printed to console)
DATA_DIR.mkdir(parents=True, exist_ok=True)
_llm_logger = logging.getLogger("brain.llm_responses")
if not _llm_logger.handlers:
    _fh = logging.FileHandler(DATA_DIR / "llm_responses.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _llm_logger.addHandler(_fh)
    _llm_logger.propagate = False
_llm_logger.setLevel(logging.DEBUG)

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
    )


def extract_from_chunk(
    messages: list[str],
    log_label: str = "",
) -> tuple[list[PatternCard], int]:
    """返回 (patterns, total_tokens)。total_tokens=0 表示 API 未返回用量信息。"""
    numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))
    prompt = _EXTRACT_PROMPT + numbered

    response = _call_llm(prompt)
    content = response.choices[0].message.content.strip()

    usage = response.usage
    total_tokens: int = usage.total_tokens if usage else 0
    prompt_tokens: int = usage.prompt_tokens if usage else 0
    completion_tokens: int = usage.completion_tokens if usage else 0

    # 写完整回复到后台日志
    _llm_logger.debug(
        "\n[%s]\n"
        "--- PROMPT (%d chars, %d tokens) ---\n%s\n"
        "--- RESPONSE (%d tokens) ---\n%s\n"
        "--- USAGE: prompt=%d completion=%d total=%d ---\n%s",
        log_label,
        len(prompt),
        prompt_tokens,
        prompt,
        completion_tokens,
        content,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        "-" * 80,
    )

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        raw_patterns = json.loads(content)
    except json.JSONDecodeError:
        return [], total_tokens

    if not isinstance(raw_patterns, list):
        return [], total_tokens

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
    return cards, total_tokens


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
