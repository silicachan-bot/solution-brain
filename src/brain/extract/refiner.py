from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime

from openai import OpenAI

import tempfile

import lancedb
import pyarrow as pa
from rich.console import Console

from brain.config import DATA_DIR, LLM_API_BASE, LLM_API_KEY, LLM_MODEL, DEDUP_TOP_N, EMBED_DIMENSIONS
from brain.models import PatternCard, FrequencyProfile

# 模块级单例，避免每次调用重建连接池
_client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
_console = Console()

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
  "template": "含 [A] [B] 占位符的模板句",
  "examples": ["2-5个真实例句"],
  "description": "模式描述：是什么、什么时候用、传达什么感觉"
}

如果这批评论中没有值得收录的模式，返回空数组 []。
只输出 JSON，不要其他文字。

评论：
"""


def _call_llm_streaming(
    prompt: str,
    on_token: Callable[[int], None] | None = None,
) -> tuple[str, int, int]:
    """流式调用 LLM，返回 (content, prompt_tokens, completion_tokens)。
    on_token(n) 在每个 token 生成后被调用，n 为当前累计 completion token 数。
    """
    stream = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    )

    parts: list[str] = []
    token_count = 0
    prompt_tokens = 0
    completion_tokens = 0

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
            token_count += 1
            if on_token:
                on_token(token_count)
        # 最后一个 chunk 含 usage（需要 stream_options include_usage）
        if getattr(chunk, "usage", None):
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or token_count

    if not completion_tokens:
        completion_tokens = token_count

    return "".join(parts), prompt_tokens, completion_tokens


_DEDUP_JUDGE_PROMPT = """\
以下是一个从 B 站评论中提取的语言模式（当前模式），以及若干个从数据库中检索到的相似候选。
请判断候选中是否有与当前模式描述同一种语言模式的条目（语义等价或高度相似，可以合并为一条记录）。

【当前模式】
模板: {current_template}
描述: {current_desc}
例句: {current_examples}

【相似候选】
{candidates_block}

如果候选中有重复的，请选出最佳匹配（只选一个）。如果都不重复，输出 0。

输出 JSON:
{{
  "duplicate_of": 0,
  "keep_description": "current" 或 "candidate",
  "reason": "一句话说明"
}}

duplicate_of: 候选编号（1 开始），0 表示无重复。
keep_description: 哪一方的描述更完整准确，仅在有重复时有意义。
只输出 JSON，不要其他文字。
"""


def _judge_duplicate_topn(
    card: PatternCard,
    candidates: list[PatternCard],
) -> tuple[int | None, str]:
    """询问 LLM 候选中是否有和 card 重复的模式。
    返回 (candidate_index_0based | None, keep_description: 'current'|'candidate')。
    解析失败或无重复返回 (None, 'current')。
    """
    if not candidates:
        return None, "current"

    parts = []
    for i, c in enumerate(candidates, 1):
        parts.append(
            f"--- 候选 {i} ({c.id}) ---\n"
            f"模板: {c.template}\n"
            f"描述: {c.description}\n"
            f"例句: {' / '.join(c.examples[:3])}"
        )
    candidates_block = "\n".join(parts)

    prompt = _DEDUP_JUDGE_PROMPT.format(
        current_template=card.template,
        current_desc=card.description,
        current_examples=" / ".join(card.examples[:3]),
        candidates_block=candidates_block,
    )
    content, _, _ = _call_llm_streaming(prompt)
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    try:
        result = json.loads(content)
        dup_of = int(result.get("duplicate_of", 0))
        keep = str(result.get("keep_description", "current"))
        if dup_of < 1 or dup_of > len(candidates):
            return None, "current"
        return dup_of - 1, keep
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, "current"


def extract_from_chunk(
    messages: list[str],
    log_label: str = "",
    on_token: Callable[[int], None] | None = None,
) -> tuple[list[PatternCard], int]:
    """返回 (patterns, total_tokens)。total_tokens=0 表示 API 未返回用量信息。"""
    numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))
    prompt = _EXTRACT_PROMPT + numbered

    content, prompt_tokens, completion_tokens = _call_llm_streaming(prompt, on_token=on_token)
    content = content.strip()
    total_tokens = prompt_tokens + completion_tokens

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
        if not all(k in p for k in ("template", "examples", "description")):
            continue
        cards.append(
            PatternCard(
                id=f"pat-{uuid.uuid4().hex[:8]}",
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


def _merge_hits(
    hits_a: list[tuple[PatternCard, float]],
    hits_b: list[tuple[PatternCard, float]],
) -> list[PatternCard]:
    """合并两路检索结果，按 card.id 去重。"""
    seen: dict[str, PatternCard] = {}
    for card, _ in hits_a + hits_b:
        if card.id not in seen:
            seen[card.id] = card
    return list(seen.values())


def _dedup_intra_batch(
    cards: list[PatternCard],
    embedder,
    top_n: int = DEDUP_TOP_N,
) -> list[PatternCard]:
    """批次内去重：用临时 LanceDB 表做双路检索（vec_template + vec_semantic），
    合并候选后交 LLM 判重。日志并列输出两路结果。
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = lancedb.connect(tmp_dir)
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("json", pa.string()),
            pa.field("vec_template", pa.list_(pa.float32(), EMBED_DIMENSIONS)),
            pa.field("vec_semantic", pa.list_(pa.float32(), EMBED_DIMENSIONS)),
        ])
        tmp_table = None
        kept: dict[str, PatternCard] = {}

        for card in cards:
            t0 = time.perf_counter()
            vec_t = embedder.embed([card.template])[0]
            vec_s = embedder.embed([card.embed_text()])[0]
            t_embed = time.perf_counter() - t0

            row = {
                "id": card.id,
                "json": json.dumps(card.to_dict(), ensure_ascii=False),
                "vec_template": vec_t,
                "vec_semantic": vec_s,
            }

            if tmp_table is None:
                tmp_table = tmp_db.create_table("batch", [row], schema=schema)
                kept[card.id] = card
                _console.print(
                    f"批次内  [cyan]{card.template!r}[/cyan]"
                    f"  [dim]embed {t_embed*1000:.0f}ms[/dim]  [dim]首条，直接入库[/dim]"
                )
                continue

            n = min(top_n, tmp_table.count_rows())

            # 双路检索
            hits_tmpl = [
                (PatternCard.from_dict(json.loads(r["json"])), 1.0 - r["_distance"])
                for r in tmp_table.search(vec_t, vector_column_name="vec_template")
                    .metric("cosine").limit(n).to_list()
            ]
            hits_sem = [
                (PatternCard.from_dict(json.loads(r["json"])), 1.0 - r["_distance"])
                for r in tmp_table.search(vec_s, vector_column_name="vec_semantic")
                    .metric("cosine").limit(n).to_list()
            ]
            merged = _merge_hits(hits_tmpl, hits_sem)

            tmpl_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_tmpl)
            sem_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_sem)
            _console.print(
                f"批次内  [cyan]{card.template!r}[/cyan]"
                f"  [dim]embed {t_embed*1000:.0f}ms[/dim]"
            )
            _console.print(f"  vec_template: {tmpl_str}")
            _console.print(f"  vec_semantic: {sem_str}")
            _console.print(f"  合并候选(去重): {len(merged)}个 → LLM 判断")

            t1 = time.perf_counter()
            dup_idx, keep_desc = _judge_duplicate_topn(card, merged)
            t_llm = time.perf_counter() - t1

            if dup_idx is not None:
                matched = merged[dup_idx]
                target = kept[matched.id]
                if keep_desc == "current":
                    target.description = card.description
                _merge_into(target, card)
                # 更新临时表中的记录
                new_vec_s = embedder.embed([target.embed_text()])[0]
                tmp_table.merge_insert("id") \
                    .when_matched_update_all() \
                    .when_not_matched_insert_all() \
                    .execute([{
                        "id": target.id,
                        "json": json.dumps(target.to_dict(), ensure_ascii=False),
                        "vec_template": embedder.embed([target.template])[0],
                        "vec_semantic": new_vec_s,
                    }])
                _console.print(
                    f"  [green]→ LLM ({t_llm:.1f}s): 与 {matched.template!r} 重复"
                    f" · 保留描述={'当前' if keep_desc == 'current' else '候选'}"
                    f" · 合并[/green]"
                )
                _console.print()
                continue

            _console.print(f"  [dim]→ LLM ({t_llm:.1f}s): 无重复，保留[/dim]")
            _console.print()
            tmp_table.add([row])
            kept[card.id] = card

    return list(kept.values())


def _dedup_against_db(
    cards: list[PatternCard],
    db,
    top_n: int = DEDUP_TOP_N,
) -> tuple[list[PatternCard], list[PatternCard]]:
    """入库前去重：每张新卡片对 DB 做双路检索（vec_template + vec_semantic），
    合并候选后交 LLM 判重。若多张新卡片都与同一已有卡片重复，合并到同一对象上。
    日志并列输出两路结果。
    """
    new_cards: list[PatternCard] = []
    updates_by_id: dict[str, PatternCard] = {}

    for card in cards:
        t0 = time.perf_counter()
        hits_tmpl = db.query_by_template(card.template, top_k=top_n)
        t_tmpl = time.perf_counter() - t0

        t1 = time.perf_counter()
        hits_sem = db.query_by_semantic(card.embed_text(), top_k=top_n)
        t_sem = time.perf_counter() - t1

        if not hits_tmpl and not hits_sem:
            _console.print(
                f"入库去重  [cyan]{card.template!r}[/cyan]"
                f"  [dim]embed+检索 {(t_tmpl+t_sem)*1000:.0f}ms[/dim]  [dim]无候选，直接新增[/dim]"
            )
            _console.print()
            new_cards.append(card)
            continue

        merged = _merge_hits(hits_tmpl, hits_sem)

        tmpl_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_tmpl) or "(空)"
        sem_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_sem) or "(空)"
        _console.print(
            f"入库去重  [cyan]{card.template!r}[/cyan]"
            f"  [dim]embed+检索 tmpl={t_tmpl*1000:.0f}ms sem={t_sem*1000:.0f}ms[/dim]"
        )
        _console.print(f"  vec_template: {tmpl_str}")
        _console.print(f"  vec_semantic: {sem_str}")
        _console.print(f"  合并候选(去重): {len(merged)}个 → LLM 判断")

        t2 = time.perf_counter()
        dup_idx, keep_desc = _judge_duplicate_topn(card, merged)
        t_llm = time.perf_counter() - t2

        if dup_idx is not None:
            matched = merged[dup_idx]
            target = updates_by_id.get(matched.id, matched)
            if keep_desc == "current":
                target.description = card.description
            _merge_into(target, card)
            updates_by_id[target.id] = target
            _console.print(
                f"  [green]→ LLM ({t_llm:.1f}s): 与 {matched.template!r} ({matched.id}) 重复"
                f" · 保留描述={'当前' if keep_desc == 'current' else '候选'}"
                f" · 更新已有[/green]"
            )
            _console.print()
            continue

        _console.print(f"  [dim]→ LLM ({t_llm:.1f}s): 无重复，新增[/dim]")
        _console.print()
        new_cards.append(card)

    return new_cards, list(updates_by_id.values())


def deduplicate_and_merge(
    cards: list[PatternCard],
    db,
    embedder,
    top_n: int = DEDUP_TOP_N,
) -> tuple[list[PatternCard], list[PatternCard]]:
    """两阶段去重：批次内相互去重 → 入库前与 DB 比对。
    使用 LanceDB 双向量列（vec_template + vec_semantic）做双路检索。
    返回 (new_cards, updated_existing_cards)。
    """
    _console.print(f"\n[bold]开始去重[/bold]: {len(cards)} 个待入库模式，top_n={top_n}")

    deduped = _dedup_intra_batch(cards, embedder, top_n)
    _console.print(f"\n批次内去重完成: {len(cards)} → [bold]{len(deduped)}[/bold] 个\n")

    new_cards, updates = _dedup_against_db(deduped, db, top_n)
    _console.print(
        f"\n去重完成: 新增 [green]{len(new_cards)}[/green] 个,"
        f" 更新 [yellow]{len(updates)}[/yellow] 个"
    )

    return new_cards, updates


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
