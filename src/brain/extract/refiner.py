from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime

from openai import OpenAI
from tqdm import tqdm

from brain.config import (
    DATA_DIR,
    DEDUP_AUTO_MERGE_THRESHOLD,
    DEDUP_SIMILARITY_THRESHOLD,
    DEDUP_TOP_N,
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


def _filter_hits_by_similarity(
    hits: list[tuple[PatternCard, float]],
    threshold: float,
) -> list[tuple[PatternCard, float]]:
    return [(card, score) for card, score in hits if score >= threshold]


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


def _find_best_hit(
    hits_a: list[tuple[PatternCard, float]],
    hits_b: list[tuple[PatternCard, float]],
) -> tuple[PatternCard, float] | None:
    """从两路命中中找出相似度最高的候选（按 card.id 去重取最大）。"""
    best: dict[str, tuple[PatternCard, float]] = {}
    for card, score in hits_a + hits_b:
        if card.id not in best or score > best[card.id][1]:
            best[card.id] = (card, score)
    if not best:
        return None
    return max(best.values(), key=lambda x: x[1])


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0


def _search_accepted(
    vec_t: list[float],
    vec_s: list[float],
    accepted: list[PatternCard],
    accepted_vecs_t: list[list[float]],
    accepted_vecs_s: list[list[float]],
    top_n: int,
) -> list[tuple[PatternCard, float]]:
    """在已接受的卡片中做 in-memory 余弦检索，返回 (card, best_sim) 列表。"""
    best: dict[str, tuple[PatternCard, float]] = {}
    for card, vt, vs in zip(accepted, accepted_vecs_t, accepted_vecs_s):
        sim = max(_cosine_sim(vec_t, vt), _cosine_sim(vec_s, vs))
        if card.id not in best or sim > best[card.id][1]:
            best[card.id] = (card, sim)
    return sorted(best.values(), key=lambda x: x[1], reverse=True)[:top_n]


def _resolve_target(
    matched: PatternCard,
    accepted: list[PatternCard],
    updates_by_id: dict[str, PatternCard],
) -> PatternCard:
    """找到可以原地修改的 target 对象；DB 来源的首次命中会加入 updates_by_id。"""
    if matched.id in updates_by_id:
        return updates_by_id[matched.id]
    for c in accepted:
        if c.id == matched.id:
            return c
    updates_by_id[matched.id] = matched
    return matched


def _refresh_accepted_vectors(
    target: PatternCard,
    accepted: list[PatternCard],
    accepted_vecs_t: list[list[float]],
    accepted_vecs_s: list[list[float]],
    embedder,
) -> None:
    """若 target 属于本批新增卡片，合并后同步刷新其缓存向量。"""
    for i, accepted_card in enumerate(accepted):
        if accepted_card.id != target.id:
            continue
        accepted_vecs_t[i] = embedder.embed([target.template])[0]
        accepted_vecs_s[i] = embedder.embed([target.embed_text()])[0]
        return


def _dedup_single_pass(
    cards: list[PatternCard],
    db,
    embedder,
    top_n: int = DEDUP_TOP_N,
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    auto_merge_threshold: float = DEDUP_AUTO_MERGE_THRESHOLD,
) -> tuple[list[PatternCard], list[PatternCard], int]:
    """单阶段去重：批量 embed 后，逐张同时检索 DB 和本批已接受卡片。
    返回 (new_cards, updated_existing_cards, total_tokens)。
    """
    if not cards:
        return [], [], 0

    total_tokens = 0

    # 1. 批量 embed（2 次 API 调用处理整批）
    vecs_t = embedder.embed([c.template for c in cards])
    vecs_s = embedder.embed([c.embed_text() for c in cards])

    accepted: list[PatternCard] = []
    accepted_vecs_t: list[list[float]] = []
    accepted_vecs_s: list[list[float]] = []
    updates_by_id: dict[str, PatternCard] = {}

    pbar = tqdm(
        zip(cards, vecs_t, vecs_s),
        total=len(cards),
        desc="去重",
        unit="条",
        leave=False,
    )
    for card, vec_t, vec_s in pbar:
        pbar.set_description(f"去重 {card.template[:18]!r}")

        # 2. 检索 DB
        hits_db_t = _filter_hits_by_similarity(
            db.query_by_vec(vec_t, "vec_template", top_n), similarity_threshold
        )
        hits_db_s = _filter_hits_by_similarity(
            db.query_by_vec(vec_s, "vec_semantic", top_n), similarity_threshold
        )

        # 3. 检索本批已接受卡片（in-memory）
        hits_acc = _filter_hits_by_similarity(
            _search_accepted(vec_t, vec_s, accepted, accepted_vecs_t, accepted_vecs_s, top_n),
            similarity_threshold,
        )

        if not hits_db_t and not hits_db_s and not hits_acc:
            tqdm.write(f"  {card.template!r}  无候选，新增")
            accepted.append(card)
            accepted_vecs_t.append(vec_t)
            accepted_vecs_s.append(vec_s)
            pbar.set_postfix(new=len(accepted), upd=len(updates_by_id))
            continue

        db_str  = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_db_t + hits_db_s) or "(空)"
        acc_str = ", ".join(f"{c.template!r}({s:.3f})" for c, s in hits_acc) or "(空)"
        tqdm.write(f"  {card.template!r}")
        tqdm.write(f"    db:  {db_str}")
        if hits_acc:
            tqdm.write(f"    acc: {acc_str}")

        # 4. 高相似度自动合并，跳过 LLM
        best_hit = _find_best_hit(hits_db_t + hits_db_s, hits_acc)
        if best_hit and best_hit[1] >= auto_merge_threshold:
            matched, best_score = best_hit
            target = _resolve_target(matched, accepted, updates_by_id)
            _merge_into(target, card)
            new_desc, enrich_tokens = _enrich_description(target, card)
            target.description = new_desc
            _refresh_accepted_vectors(target, accepted, accepted_vecs_t, accepted_vecs_s, embedder)
            total_tokens += enrich_tokens
            tqdm.write(f"    → 相似度 {best_score:.3f} ≥ {auto_merge_threshold}，自动合并 {matched.template!r}")
            pbar.set_postfix(new=len(accepted), upd=len(updates_by_id))
            continue

        # 5. LLM 判断
        candidates = _merge_hits(hits_db_t + hits_db_s, hits_acc)
        tqdm.write(f"    → 候选 {len(candidates)} 个，LLM 判断中…")
        t0 = time.perf_counter()
        dup_idx, judge_tokens = _judge_duplicate_topn(card, candidates)
        total_tokens += judge_tokens
        t_llm = time.perf_counter() - t0

        if dup_idx is not None:
            matched = candidates[dup_idx]
            target = _resolve_target(matched, accepted, updates_by_id)
            _merge_into(target, card)
            new_desc, enrich_tokens = _enrich_description(target, card)
            target.description = new_desc
            _refresh_accepted_vectors(target, accepted, accepted_vecs_t, accepted_vecs_s, embedder)
            total_tokens += enrich_tokens
            tqdm.write(f"    → LLM {t_llm:.1f}s: 与 {matched.template!r} 重复，合并")
            pbar.set_postfix(new=len(accepted), upd=len(updates_by_id))
            continue

        tqdm.write(f"    → LLM {t_llm:.1f}s: 无重复，新增")
        accepted.append(card)
        accepted_vecs_t.append(vec_t)
        accepted_vecs_s.append(vec_s)
        pbar.set_postfix(new=len(accepted), upd=len(updates_by_id))

    return accepted, list(updates_by_id.values()), total_tokens


def deduplicate_and_merge(
    cards: list[PatternCard],
    db,
    embedder,
    top_n: int = DEDUP_TOP_N,
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    auto_merge_threshold: float = DEDUP_AUTO_MERGE_THRESHOLD,
) -> tuple[list[PatternCard], list[PatternCard], int]:
    """单阶段去重：批量 embed + 同时检索 DB 和本批已接受卡片。
    返回 (new_cards, updated_existing_cards, total_dedup_tokens)。
    """
    tqdm.write(
        f"\n开始去重: {len(cards)} 个待入库模式，"
        f"top_n={top_n}, threshold={similarity_threshold:.2f}, auto_merge≥{auto_merge_threshold:.2f}"
    )

    new_cards, updates, total_tokens = _dedup_single_pass(
        cards, db, embedder, top_n, similarity_threshold, auto_merge_threshold
    )
    tqdm.write(f"去重完成: 新增 {len(new_cards)} 个, 更新 {len(updates)} 个  ({total_tokens} tokens)")

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
