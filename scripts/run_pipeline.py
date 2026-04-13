"""
全流程提取 pipeline：摄入 → 提取 → 存储

用法:
    uv run python scripts/run_pipeline.py                    # 增量处理
    uv run python scripts/run_pipeline.py --full             # 全量重跑
    uv run python scripts/run_pipeline.py --limit 5          # 限制处理N个视频
    uv run python scripts/run_pipeline.py --chunk-size 30    # 每块评论数
    uv run python scripts/run_pipeline.py --dry-run          # 只看计划不调LLM
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm import tqdm

from brain.config import BILIBILI_DB_PATH, LANCEDB_DIR, CHUNK_SIZE, STATE_FILE
from brain.ingest.reader import BilibiliReader
from brain.ingest.cleaner import clean_comments
from brain.ingest.state import WatermarkState
from brain.extract.chunker import build_comment_pairs, chunk_comment_pairs, chunk_comments
from brain.extract.refiner import extract_from_chunk, deduplicate_and_merge
from brain.store.pattern_db import PatternDB
from brain.store.embedding import QwenEmbedder


def main():
    parser = argparse.ArgumentParser(description="运行语言模式提取 pipeline")
    parser.add_argument("--full", action="store_true", help="全量重跑（忽略水位线）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理N个视频（0=全部）")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="每块评论数")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不调用LLM")
    parser.add_argument("--max-chunks", type=int, default=0, help="每个视频最多处理N块（0=全部）")
    args = parser.parse_args()

    reader = BilibiliReader(BILIBILI_DB_PATH)
    state = WatermarkState(STATE_FILE)
    embedder = QwenEmbedder()
    db = PatternDB(LANCEDB_DIR, embedder=embedder)

    # 1. 列出视频
    videos = reader.list_videos()
    tqdm.write(f"数据库中已完成爬取的视频: {len(videos)} 个")

    # 2. 水位线过滤（增量）
    watermark = None if args.full else state.get_watermark("bilibili")
    if watermark:
        videos = [v for v in videos if v["bvid"] > watermark]
        tqdm.write(f"水位线过滤后: {len(videos)} 个新视频")

    if args.limit > 0:
        videos = videos[: args.limit]
        tqdm.write(f"限制处理: {len(videos)} 个视频")

    if not videos:
        tqdm.write("没有需要处理的视频。")
        return

    # dry-run
    if args.dry_run:
        for i, video in enumerate(videos, 1):
            bvid = video["bvid"]
            comments = reader.read_comments(bvid)
            cleaned = clean_comments(comments)
            comment_pairs = build_comment_pairs(cleaned)
            chunks = chunk_comments(comment_pairs, chunk_size=args.chunk_size)
            if args.max_chunks > 0:
                chunks = chunks[: args.max_chunks]
            print(f"[{i}/{len(videos)}] {bvid} — {video.get('title','无标题')}")
            print(
                f"  评论: {len(comments)} → 清洗后: {len(cleaned)} "
                f"→ 评论对: {len(comment_pairs)} → {len(chunks)} 块 [dry-run]"
            )
        return

    # 3. 逐视频处理
    extract_tokens = 0
    total_patterns = 0
    all_patterns: list = []

    video_pbar = tqdm(videos, desc="视频", unit="个", position=0)
    for video in video_pbar:
        bvid = video["bvid"]
        title = video.get("title", "无标题")
        video_pbar.set_description(f"视频 {bvid}")

        comments = reader.read_comments(bvid)
        cleaned = clean_comments(comments)
        comment_pairs = build_comment_pairs(cleaned)

        if not comment_pairs:
            video_pbar.set_postfix(pairs=0)
            continue

        pair_chunks = chunk_comment_pairs(comment_pairs, chunk_size=args.chunk_size)
        chunks = chunk_comments(comment_pairs, chunk_size=args.chunk_size)
        if args.max_chunks > 0:
            pair_chunks = pair_chunks[: args.max_chunks]
            chunks = chunks[: args.max_chunks]

        video_tokens = 0
        video_patterns: list = []

        chunk_pbar = tqdm(
            zip(chunks, pair_chunks),
            total=len(chunks),
            desc="  分块",
            unit="块",
            position=1,
            leave=False,
        )
        for j, (chunk, pair_chunk) in enumerate(chunk_pbar, 1):
            streaming_tokens = 0
            chunk_pbar.set_description(f"  分块 {j}/{len(chunks)}")

            def on_token(n: int, _pbar=chunk_pbar) -> None:
                nonlocal streaming_tokens
                streaming_tokens = n
                if n % 30 == 0:
                    _pbar.set_postfix(gen=n)

            patterns, tokens = extract_from_chunk(
                chunk,
                log_label=f"{bvid} chunk {j}/{len(chunks)}",
                on_token=on_token,
                comment_pairs=pair_chunk,
                video_title=title,
            )
            video_tokens += tokens
            extract_tokens += tokens
            video_patterns.extend(patterns)
            total_patterns += len(patterns)
            all_patterns.extend(patterns)
            chunk_pbar.set_postfix(tok=tokens, pat=len(patterns))

        tqdm.write(
            f"  {bvid}  {len(chunks)} 块  {video_tokens:,} tokens  {len(video_patterns)} 个模式  {title[:30]}"
        )
        video_pbar.set_postfix(
            tok=f"{extract_tokens/1e6:.3f}M",
            pat=total_patterns,
        )

    video_pbar.close()

    # 4. 去重合并
    tqdm.write(f"\n提取到的原始模式: {len(all_patterns)} 个")
    tqdm.write(f"数据库中已有模式: {db.count()} 个")

    new_cards, updates, dedup_tokens = deduplicate_and_merge(all_patterns, db, embedder)

    db.save(new_cards)
    db.update(updates)

    # 5. 更新水位线
    if videos:
        last_bvid = videos[-1]["bvid"]
        state.set_watermark("bilibili", last_bvid)
        tqdm.write(f"水位线更新至: {last_bvid}")

    total_tokens = extract_tokens + dedup_tokens
    tqdm.write(
        f"\n完成。数据库中模式总数: {db.count()}"
        f"\n共消耗 {total_tokens/1e6:.3f} M tokens"
        f"  （提取 {extract_tokens/1e6:.3f} M + 去重 {dedup_tokens/1e6:.3f} M）"
    )


if __name__ == "__main__":
    main()
