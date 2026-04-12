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
from tqdm import tqdm

from brain.config import (
    DATA_DIR,
    DEDUP_SIMILARITY_THRESHOLD,
    DEDUP_TOP_N,
    EMBED_DIMENSIONS,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)
from brain.models import CommentPair, FrequencyProfile, PatternCard, PatternOrigin
from brain.prompts import render_prompt

# 模块级单例，避免每次调用重建连接池
_client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT_SECONDS)

# File logger for full LLM responses (background log, not printed to console)
DATA_DIR.mkdir(parents=True, exist_ok=True)
_llm_logger = logging.getLogger("brain.llm_responses")
if not _llm_logger.handlers:
    _fh = logging.FileHandler(DATA_DIR / "llm_responses.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _llm_logger.addHandler(_fh)
    _llm_logger.propagate = False
_llm_logger.setLevel(logging.DEBUG)


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


def _judge_duplicate_topn(
    card: PatternCard,
    candidates: list[PatternCard],
) -> tuple[int | None, int]:
    """询问 LLM 候选中是否有和 card 重复的模式。
    返回 (candidate_index_0based | None, total_tokens)。
    解析失败或无重复返回 (None, tokens)。
    """
    if not candidates:
        return None, 0

    parts = []
    for i, c in enumerate(candidates, 1):
        parts.append(
            f"--- 候选 {i} ({c.id}) ---\n"
            f"模板: {c.template}\n"
            f"描述: {c.description}\n"
            f"例句: {' / '.join(c.examples[:3])}"
        )
    candidates_block = "\n".join(parts)

    prompt = render_prompt(
        "extract_dedup_judge.txt",
        current_template=card.template,
        current_desc=card.description,
        current_examples=" / ".join(card.examples[:3]),
        candidates_block=candidates_block,
    )
    content, prompt_tokens, completion_tokens = _call_llm_streaming(prompt)
    total_tokens = prompt_tokens + completion_tokens
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    try:
        result = json.loads(content)
        dup_of = int(result.get("duplicate_of", 0))
        if dup_of < 1 or dup_of > len(candidates):
            return None, total_tokens
        return dup_of - 1, total_tokens
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, total_tokens


def _build_origins_for_examples(
    examples: list[str],
    comment_pairs: list[CommentPair] | None,
    video_title: str,
) -> list[PatternOrigin]:
    if not comment_pairs:
        return []

    origins: list[PatternOrigin] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for example in examples:
        for pair in comment_pairs:
            if pair.reply.message != example:
                continue
            origin = PatternOrigin(
                example=example,
                bvid=pair.reply.bvid,
                video_title=video_title,
                parent_message=pair.parent.message,
                reply_message=pair.reply.message,
            )
            key = (
                origin.example,
                origin.bvid,
                origin.video_title,
                origin.parent_message,
                origin.reply_message,
            )
            if key in seen:
                continue
            seen.add(key)
            origins.append(origin)

    return origins


def extract_from_chunk(
    messages: list[str],
    log_label: str = "",
    on_token: Callable[[int], None] | None = None,
    *,
    comment_pairs: list[CommentPair] | None = None,
    video_title: str = "",
) -> tuple[list[PatternCard], int]:
    """返回 (patterns, total_tokens)。total_tokens=0 表示 API 未返回用量信息。"""
    numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))
    prompt = render_prompt("extract_patterns.txt", comments=numbered)

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
        examples = p["examples"][:5]
        cards.append(
            PatternCard(
                id=f"pat-{uuid.uuid4().hex[:8]}",
                description=p["description"],
                template=p["template"],
                examples=examples,
                frequency=FrequencyProfile(recent=1, medium=1, long_term=1, total=1),
                source="bilibili",
                created_at=now,
                updated_at=now,
                origins=_build_origins_for_examples(examples, comment_pairs, video_title),
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


def _filter_hits_by_similarity(
    hits: list[tuple[PatternCard, float]],
    threshold: float,
) -> list[tuple[PatternCard, float]]:
    """仅保留达到相似度阈值的候选。"""
    return [(card, score) for card, score in hits if score >= threshold]


def _dedup_intra_batch(
    cards: list[PatternCard],
    embedder,
    top_n: int = DEDUP_TOP_N,
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> tuple[list[PatternCard], int]:
    """批次内去重：双路向量检索 + LLM 判重。
    返回 (kept_cards, total_tokens)。
    """
    total_tokens = 0
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

        pbar = tqdm(cards, desc="批次内去重", unit="条", leave=False)
        for card in pbar:
            pbar.set_description(f"批次内去重 {card.template[:18]!r}")
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
                tqdm.write(f"  批次内 {card.template!r}  embed {t_embed*1000:.0f}ms  首条入库")
                pbar.set_postfix(kept=len(kept), tokens=f"{total_tokens/1000:.1f}k")
                continue

            n = min(top_n, tmp_table.count_rows())
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
            filtered_tmpl = _filter_hits_by_similarity(hits_tmpl, similarity_threshold)
            filtered_sem = _filter_hits_by_similarity(hits_sem, similarity_threshold)
            merged = _merge_hits(filtered_tmpl, filtered_sem)

            tmpl_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_tmpl)
            sem_str  = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_sem)
            tqdm.write(f"  批次内 {card.template!r}  embed {t_embed*1000:.0f}ms")
            tqdm.write(f"    tmpl: {tmpl_str}")
            tqdm.write(f"    sem:  {sem_str}")

            if not merged:
                tqdm.write(f"    → 阈值过滤后无候选，直接保留")
                tmp_table.add([row])
                kept[card.id] = card
                pbar.set_postfix(kept=len(kept), tokens=f"{total_tokens/1000:.1f}k")
                continue

            tqdm.write(f"    → 候选 {len(merged)} 个，LLM 判断中…")
            t1 = time.perf_counter()
            dup_idx, judge_tokens = _judge_duplicate_topn(card, merged)
            total_tokens += judge_tokens
            t_llm = time.perf_counter() - t1

            if dup_idx is not None:
                matched = merged[dup_idx]
                target = kept[matched.id]
                _merge_into(target, card)
                new_desc, enrich_tokens = _enrich_description(target, card)
                target.description = new_desc
                total_tokens += enrich_tokens
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
                tqdm.write(
                    f"    → LLM {t_llm:.1f}s: 与 {matched.template!r} 重复，例句合并+描述增补"
                )
                pbar.set_postfix(kept=len(kept), tokens=f"{total_tokens/1000:.1f}k")
                continue

            tqdm.write(f"    → LLM {t_llm:.1f}s: 无重复，保留")
            tmp_table.add([row])
            kept[card.id] = card
            pbar.set_postfix(kept=len(kept), tokens=f"{total_tokens/1000:.1f}k")

    return list(kept.values()), total_tokens


def _dedup_against_db(
    cards: list[PatternCard],
    db,
    top_n: int = DEDUP_TOP_N,
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> tuple[list[PatternCard], list[PatternCard], int]:
    """入库前去重：双路向量检索 + LLM 判重。
    返回 (new_cards, updates, total_tokens)。
    """
    total_tokens = 0
    new_cards: list[PatternCard] = []
    updates_by_id: dict[str, PatternCard] = {}

    pbar = tqdm(cards, desc="入库去重", unit="条", leave=False)
    for card in pbar:
        pbar.set_description(f"入库去重 {card.template[:18]!r}")
        t0 = time.perf_counter()
        hits_tmpl = db.query_by_template(card.template, top_k=top_n)
        t_tmpl = time.perf_counter() - t0

        t1 = time.perf_counter()
        hits_sem = db.query_by_semantic(card.embed_text(), top_k=top_n)
        t_sem = time.perf_counter() - t1

        filtered_tmpl = _filter_hits_by_similarity(hits_tmpl, similarity_threshold)
        filtered_sem = _filter_hits_by_similarity(hits_sem, similarity_threshold)

        if not filtered_tmpl and not filtered_sem:
            tqdm.write(
                f"  入库 {card.template!r}  检索 {(t_tmpl+t_sem)*1000:.0f}ms  无候选，直接新增"
            )
            new_cards.append(card)
            pbar.set_postfix(new=len(new_cards), upd=len(updates_by_id), tokens=f"{total_tokens/1000:.1f}k")
            continue

        merged = _merge_hits(filtered_tmpl, filtered_sem)
        tmpl_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_tmpl) or "(空)"
        sem_str  = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_sem) or "(空)"
        tqdm.write(f"  入库 {card.template!r}  tmpl={t_tmpl*1000:.0f}ms sem={t_sem*1000:.0f}ms")
        tqdm.write(f"    tmpl: {tmpl_str}")
        tqdm.write(f"    sem:  {sem_str}")
        tqdm.write(f"    → 候选 {len(merged)} 个，LLM 判断中…")

        t2 = time.perf_counter()
        dup_idx, judge_tokens = _judge_duplicate_topn(card, merged)
        total_tokens += judge_tokens
        t_llm = time.perf_counter() - t2

        if dup_idx is not None:
            matched = merged[dup_idx]
            target = updates_by_id.get(matched.id, matched)
            _merge_into(target, card)
            new_desc, enrich_tokens = _enrich_description(target, card)
            target.description = new_desc
            total_tokens += enrich_tokens
            updates_by_id[target.id] = target
            tqdm.write(
                f"    → LLM {t_llm:.1f}s: 与 {matched.template!r} ({matched.id}) 重复，更新已有"
            )
            pbar.set_postfix(new=len(new_cards), upd=len(updates_by_id), tokens=f"{total_tokens/1000:.1f}k")
            continue

        tqdm.write(f"    → LLM {t_llm:.1f}s: 无重复，新增")
        new_cards.append(card)
        pbar.set_postfix(new=len(new_cards), upd=len(updates_by_id), tokens=f"{total_tokens/1000:.1f}k")

    return new_cards, list(updates_by_id.values()), total_tokens


def deduplicate_and_merge(
    cards: list[PatternCard],
    db,
    embedder,
    top_n: int = DEDUP_TOP_N,
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> tuple[list[PatternCard], list[PatternCard], int]:
    """两阶段去重：批次内相互去重 → 入库前与 DB 比对。
    返回 (new_cards, updated_existing_cards, total_dedup_tokens)。
    """
    tqdm.write(
        f"\n开始去重: {len(cards)} 个待入库模式，"
        f"top_n={top_n}, threshold={similarity_threshold:.2f}"
    )

    deduped, intra_tokens = _dedup_intra_batch(cards, embedder, top_n, similarity_threshold)
    tqdm.write(f"批次内去重完成: {len(cards)} → {len(deduped)} 个  ({intra_tokens} tokens)")

    new_cards, updates, db_tokens = _dedup_against_db(deduped, db, top_n, similarity_threshold)
    total_tokens = intra_tokens + db_tokens
    tqdm.write(
        f"去重完成: 新增 {len(new_cards)} 个, 更新 {len(updates)} 个  ({db_tokens} tokens)"
    )

    return new_cards, updates, total_tokens


def _enrich_description(target: PatternCard, incoming: PatternCard) -> tuple[str, int]:
    """基于原有描述 + 新增描述 + 全量例句，调用 LLM 生成更丰富的结构化描述。
    返回 (description, total_tokens)。LLM 调用失败时回退到原有描述。
    """
    examples_text = "\n".join(f"- {e}" for e in target.examples)
    prompt = render_prompt(
        "extract_merge_description.txt",
        template=target.template,
        existing_description=target.description,
        new_description=incoming.description,
        examples=examples_text,
    )
    content, prompt_tokens, completion_tokens = _call_llm_streaming(prompt)
    total_tokens = prompt_tokens + completion_tokens
    content = content.strip()
    return (content if content else target.description), total_tokens


def _merge_into(target: PatternCard, source: PatternCard) -> None:
    """合并 source 的频率统计、例句和来源到 target，全量保留例句（不设上限）。"""
    target.frequency.recent += source.frequency.recent
    target.frequency.medium += source.frequency.medium
    target.frequency.long_term += source.frequency.long_term
    target.frequency.total += source.frequency.total

    merged_examples: list[str] = []
    seen_examples: set[str] = set()
    for example in target.examples + source.examples:
        if example in seen_examples:
            continue
        seen_examples.add(example)
        merged_examples.append(example)

    merged_origins: list[PatternOrigin] = []
    seen_origins: set[tuple[str, str, str, str, str]] = set()
    for example in merged_examples:
        for origin in target.origins + source.origins:
            if origin.example != example:
                continue
            key = (
                origin.example,
                origin.bvid,
                origin.video_title,
                origin.parent_message,
                origin.reply_message,
            )
            if key in seen_origins:
                continue
            seen_origins.add(key)
            merged_origins.append(origin)

    target.examples = merged_examples
    target.origins = merged_origins

    target.updated_at = datetime.now()
