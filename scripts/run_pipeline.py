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

from brain.config import BILIBILI_DB_PATH, CHROMA_DIR, CHUNK_SIZE, STATE_FILE
from brain.ingest.reader import BilibiliReader
from brain.ingest.cleaner import clean_comments
from brain.ingest.state import WatermarkState
from brain.extract.chunker import chunk_comments
from brain.extract.refiner import extract_from_chunk, deduplicate_and_merge
from brain.store.pattern_db import PatternDB
from brain.store.embedding import QwenEmbeddingFunction


def main():
    parser = argparse.ArgumentParser(description="运行语言模式提取 pipeline")
    parser.add_argument("--full", action="store_true", help="全量重跑（忽略水位线）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理N个视频（0=全部）")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="每块评论数")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不调用LLM")
    args = parser.parse_args()

    reader = BilibiliReader(BILIBILI_DB_PATH)
    state = WatermarkState(STATE_FILE)
    db = PatternDB(CHROMA_DIR, embedding_fn=QwenEmbeddingFunction())

    # 1. 列出视频
    videos = reader.list_videos()
    print(f"数据库中已完成爬取的视频: {len(videos)} 个")

    # 2. 水位线过滤（增量）
    watermark = None if args.full else state.get_watermark("bilibili")
    if watermark:
        videos = [v for v in videos if v["bvid"] > watermark]
        print(f"水位线过滤后: {len(videos)} 个新视频")

    if args.limit > 0:
        videos = videos[: args.limit]
        print(f"限制处理: {len(videos)} 个视频")

    if not videos:
        print("没有需要处理的视频。")
        return

    # 3. 逐视频处理
    all_patterns = []
    for i, video in enumerate(videos, 1):
        bvid = video["bvid"]
        title = video.get("title", "无标题")
        print(f"\n[{i}/{len(videos)}] {bvid} — {title}")

        comments = reader.read_comments(bvid)
        print(f"  原始评论: {len(comments)} 条")

        cleaned = clean_comments(comments)
        print(f"  清洗后: {len(cleaned)} 条")

        if not cleaned:
            continue

        chunks = chunk_comments(cleaned, chunk_size=args.chunk_size)
        print(f"  分块: {len(chunks)} 块")

        if args.dry_run:
            print("  [dry-run] 跳过LLM提取")
            continue

        for j, chunk in enumerate(chunks, 1):
            print(f"  提取第 {j}/{len(chunks)} 块...", end=" ", flush=True)
            patterns = extract_from_chunk(chunk)
            print(f"发现 {len(patterns)} 个模式")
            all_patterns.extend(patterns)

    if args.dry_run:
        print(f"\n[dry-run] 共计 {sum(1 for v in videos if reader.read_comments(v['bvid']))} 个视频有评论")
        return

    # 4. 去重合并
    print(f"\n提取到的原始模式: {len(all_patterns)} 个")
    existing = db.list_all()
    print(f"数据库中已有模式: {len(existing)} 个")

    new_cards, updates = deduplicate_and_merge(all_patterns, existing)
    print(f"新模式: {len(new_cards)} 个, 更新: {len(updates)} 个")

    db.save(new_cards)
    db.update(updates)

    # 5. 更新水位线
    if videos:
        last_bvid = videos[-1]["bvid"]
        state.set_watermark("bilibili", last_bvid)
        print(f"水位线更新至: {last_bvid}")

    total = len(existing) + len(new_cards)
    print(f"\n完成。数据库中模式总数: {total}")


if __name__ == "__main__":
    main()
