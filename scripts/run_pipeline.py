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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain.config import BILIBILI_DB_PATH, LANCEDB_DIR, CHUNK_SIZE, STATE_FILE
from brain.ingest.reader import BilibiliReader
from brain.ingest.cleaner import clean_comments
from brain.ingest.state import WatermarkState
from brain.extract.chunker import chunk_comments
from brain.extract.refiner import extract_from_chunk, deduplicate_and_merge
from brain.store.pattern_db import PatternDB
from brain.store.embedding import QwenEmbedder

from rich import box
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.1f}s" if m else f"{s:.1f}s"


def main():
    parser = argparse.ArgumentParser(description="运行语言模式提取 pipeline")
    parser.add_argument("--full", action="store_true", help="全量重跑（忽略水位线）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理N个视频（0=全部）")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="每块评论数")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不调用LLM")
    args = parser.parse_args()

    reader = BilibiliReader(BILIBILI_DB_PATH)
    state = WatermarkState(STATE_FILE)
    embedder = QwenEmbedder()
    db = PatternDB(LANCEDB_DIR, embedder=embedder)

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

    # dry-run: 简单打印不需要 Rich
    if args.dry_run:
        for i, video in enumerate(videos, 1):
            bvid = video["bvid"]
            comments = reader.read_comments(bvid)
            cleaned = clean_comments(comments)
            chunks = chunk_comments(cleaned, chunk_size=args.chunk_size)
            print(f"[{i}/{len(videos)}] {bvid} — {video.get('title','无标题')}")
            print(f"  评论: {len(comments)} → 清洗后: {len(cleaned)} → {len(chunks)} 块 [dry-run]")
        return

    # 3. 逐视频处理（Rich 实时展示）
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )
    video_task = progress.add_task("[cyan]视频", total=len(videos))
    chunk_task = progress.add_task("[green]分块", total=1, visible=False)

    completed_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    completed_table.add_column("BV号", style="cyan", no_wrap=True)
    completed_table.add_column("标题", max_width=26)
    completed_table.add_column("分块", justify="right")
    completed_table.add_column("耗时", justify="right", style="green")
    completed_table.add_column("Tokens", justify="right", style="yellow")
    completed_table.add_column("模式", justify="right", style="magenta")

    pipeline_start = time.monotonic()
    total_tokens = 0
    total_patterns = 0
    all_patterns: list = []
    streaming_tokens = 0  # 当前 chunk 已生成的 token 数（流式实时）

    def make_display() -> Group:
        elapsed = time.monotonic() - pipeline_start
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        stats = Text.from_markup(
            f"[dim]总耗时[/] [green]{h:02d}:{m:02d}:{s:02d}[/]  "
            f"[dim]累计 tokens[/] [yellow]{total_tokens:,}[/]  "
            f"[dim]发现模式[/] [magenta]{total_patterns}[/]"
        )
        if streaming_tokens > 0:
            progress.update(chunk_task, description=f"[green]分块  [dim]⟳ 生成中 {streaming_tokens} tok")
        return Group(progress, stats, completed_table)

    with Live(make_display(), refresh_per_second=4, vertical_overflow="visible") as live:
        for i, video in enumerate(videos, 1):
            bvid = video["bvid"]
            title = video.get("title", "无标题")

            progress.update(video_task, description=f"[cyan]视频  {bvid} — {title[:20]}")
            live.update(make_display())

            comments = reader.read_comments(bvid)
            cleaned = clean_comments(comments)

            if not cleaned:
                progress.advance(video_task)
                live.update(make_display())
                continue

            chunks = chunk_comments(cleaned, chunk_size=args.chunk_size)

            progress.update(
                chunk_task,
                total=len(chunks),
                completed=0,
                visible=True,
                description="[green]分块",
            )
            live.update(make_display())

            video_start = time.monotonic()
            video_tokens = 0
            video_patterns: list = []

            for j, chunk in enumerate(chunks, 1):
                streaming_tokens = 0
                progress.update(chunk_task, description=f"[green]分块 {j}/{len(chunks)}")
                live.update(make_display())

                def on_token(n: int, _live=live) -> None:
                    nonlocal streaming_tokens
                    streaming_tokens = n
                    if n % 20 == 0:
                        _live.update(make_display())

                log_label = f"{bvid} chunk {j}/{len(chunks)}"
                patterns, tokens = extract_from_chunk(chunk, log_label=log_label, on_token=on_token)

                video_tokens += tokens
                total_tokens += tokens
                video_patterns.extend(patterns)
                total_patterns += len(patterns)
                all_patterns.extend(patterns)

                streaming_tokens = 0
                progress.advance(chunk_task)
                progress.update(chunk_task, description=f"[green]分块 {j}/{len(chunks)}")
                live.update(make_display())

            completed_table.add_row(
                bvid,
                (title[:25] + "…") if len(title) > 25 else title,
                str(len(chunks)),
                _fmt_elapsed(time.monotonic() - video_start),
                f"{video_tokens:,}",
                str(len(video_patterns)),
            )
            progress.advance(video_task)
            progress.update(chunk_task, visible=False)
            live.update(make_display())

    # 4. 去重合并
    print(f"\n提取到的原始模式: {len(all_patterns)} 个")
    print(f"数据库中已有模式: {db.count()} 个")

    new_cards, updates = deduplicate_and_merge(all_patterns, db, embedder)
    print(f"新模式: {len(new_cards)} 个, 更新: {len(updates)} 个")

    db.save(new_cards)
    db.update(updates)

    # 5. 更新水位线
    if videos:
        last_bvid = videos[-1]["bvid"]
        state.set_watermark("bilibili", last_bvid)
        print(f"水位线更新至: {last_bvid}")

    total = db.count()
    print(f"完成。数据库中模式总数: {total}")


if __name__ == "__main__":
    main()
