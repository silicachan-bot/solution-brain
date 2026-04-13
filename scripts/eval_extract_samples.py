from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain.config import BILIBILI_DB_PATH
from brain.ingest.cleaner import clean_comments
from brain.ingest.reader import BilibiliReader
from brain.extract.chunker import build_comment_pairs, chunk_comment_pairs, chunk_comments
from brain.extract.refiner import extract_from_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="对指定视频样本运行模式提取，输出提取结果")
    parser.add_argument("bvids", nargs="+", help="要评估的 BVID 列表")
    parser.add_argument("--chunk-size", type=int, default=50, help="每块评论对数量")
    parser.add_argument("--pair-limit", type=int, default=0, help="只评估前 N 个评论对，0 表示全部")
    args = parser.parse_args()

    reader = BilibiliReader(BILIBILI_DB_PATH)
    videos = {video["bvid"]: video for video in reader.list_videos()}

    for bvid in args.bvids:
        comments = clean_comments(reader.read_comments(bvid))
        pairs = build_comment_pairs(comments)
        if args.pair_limit > 0:
            pairs = pairs[:args.pair_limit]
        pair_chunks = chunk_comment_pairs(pairs, chunk_size=args.chunk_size)
        chunks = chunk_comments(pairs, chunk_size=args.chunk_size)
        title = videos.get(bvid, {}).get("title", "")

        print(f"===== {bvid} =====", flush=True)
        print(f"cleaned_comments={len(comments)} comment_pairs={len(pairs)} chunks={len(chunks)}", flush=True)

        all_cards = []
        total_tokens = 0
        for i, (chunk, pair_chunk) in enumerate(zip(chunks, pair_chunks, strict=False), 1):
            print(f"running chunk {i}/{len(chunks)}", flush=True)
            cards, tokens = extract_from_chunk(
                chunk,
                log_label=f"{bvid} eval chunk {i}/{len(chunks)}",
                comment_pairs=pair_chunk,
                video_title=title,
            )
            all_cards.extend(cards)
            total_tokens += tokens

        print(f"tokens={total_tokens} patterns={len(all_cards)}", flush=True)
        for idx, card in enumerate(all_cards, 1):
            print(f"{idx}. template: {card.template}", flush=True)
            print(f"   desc: {card.description}", flush=True)
            print(f"   examples: {' | '.join(card.examples)}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
