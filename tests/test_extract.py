import json
import os
from unittest.mock import patch
from datetime import datetime

import pytest

from brain.models import CleanedComment, CommentPair, PatternCard, FrequencyProfile
from brain.extract.chunker import build_comment_pairs, chunk_comments, format_comment_pair
from brain.extract.refiner import extract_from_chunk, _judge_duplicate_topn, deduplicate_and_merge
from brain.prompts import load_prompt


def _make_comment(
    rpid: int,
    message: str,
    *,
    root: int = 0,
    parent: int = 0,
) -> CleanedComment:
    return CleanedComment(
        rpid=rpid,
        bvid="BV1test",
        uid=1,
        uname=f"user-{rpid}",
        message=message,
        ctime=1700000000 + rpid,
        root=root,
        parent=parent,
    )


class TestChunker:
    def test_builds_parent_reply_pairs(self):
        comments = [
            _make_comment(1, "root"),
            _make_comment(2, "reply", root=1, parent=1),
            _make_comment(3, "nested reply", root=1, parent=2),
            _make_comment(4, "standalone"),
        ]
        pairs = build_comment_pairs(comments)

        assert len(pairs) == 2
        assert pairs[0].parent.rpid == 1
        assert pairs[0].reply.rpid == 2
        assert pairs[1].parent.rpid == 2
        assert pairs[1].reply.rpid == 3

    def test_ignores_replies_with_missing_parent(self):
        comments = [
            _make_comment(2, "reply", root=1, parent=1),
        ]
        assert build_comment_pairs(comments) == []

    def test_formats_comment_pair(self):
        pair = CommentPair(
            parent=_make_comment(1, "上文"),
            reply=_make_comment(2, "回复", root=1, parent=1),
        )
        assert format_comment_pair(pair) == "上文评论：上文\n回复评论：回复"

    def test_basic_chunking(self):
        pairs = [
            CommentPair(
                parent=_make_comment(i, f"parent {i}"),
                reply=_make_comment(i + 1000, f"reply {i}", root=i, parent=i),
            )
            for i in range(120)
        ]
        chunks = chunk_comments(pairs, chunk_size=50)
        assert len(chunks) == 3
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50
        assert len(chunks[2]) == 20

    def test_small_input(self):
        pairs = [
            CommentPair(
                parent=_make_comment(i, f"parent {i}"),
                reply=_make_comment(i + 100, f"reply {i}", root=i, parent=i),
            )
            for i in range(10)
        ]
        chunks = chunk_comments(pairs, chunk_size=50)
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_empty_input(self):
        chunks = chunk_comments([], chunk_size=50)
        assert chunks == []

    def test_returns_message_strings(self):
        pairs = [
            CommentPair(
                parent=_make_comment(1, "hello"),
                reply=_make_comment(2, "world", root=1, parent=1),
            )
        ]
        chunks = chunk_comments(pairs, chunk_size=50)
        assert chunks == [["上文评论：hello\n回复评论：world"]]


class TestExtractFromChunk:
    def test_parses_llm_response(self):
        mock_patterns = [
            {
                "template": "[A]...好家伙...",
                "examples": ["这也行...好家伙...", "又来了...好家伙..."],
                "description": "表达无奈和吐槽",
            }
        ]
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=(json.dumps(mock_patterns, ensure_ascii=False), 10, 50),
        ):
            cards, total_tokens = extract_from_chunk(["comment1", "comment2"])

        assert len(cards) == 1
        assert cards[0].template == "[A]...好家伙..."

    def test_handles_empty_response(self):
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=("[]", 5, 3),
        ):
            cards, total_tokens = extract_from_chunk(["comment1", "comment2"])

        assert cards == []
        assert total_tokens == 8

    def test_renders_numbered_comments_into_prompt(self):
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=("[]", 5, 3),
        ) as mock_call:
            extract_from_chunk([
                "上文评论：first parent\n回复评论：first reply",
                "上文评论：second parent\n回复评论：second reply",
            ])

        rendered_prompt = mock_call.call_args.args[0]
        assert "1. 上文评论：first parent\n回复评论：first reply" in rendered_prompt
        assert "2. 上文评论：second parent\n回复评论：second reply" in rendered_prompt
        assert "评论对" in rendered_prompt
        assert "{{ comments }}" not in rendered_prompt


class TestDeduplicateAndMerge:
    def _make_card(self, cid: str, template: str, description: str = "默认",
                   examples: list[str] | None = None) -> PatternCard:
        return PatternCard(
            id=cid, description=description,
            template=template, examples=examples or ["示例"],
            frequency=FrequencyProfile(1, 1, 1, 1),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_no_duplicates(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        cards = [
            self._make_card("1", "[A]不同A", description="descA", examples=["exA"]),
            self._make_card("2", "[A]不同B", description="descB", examples=["exB"]),
        ]
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, "current")):
            new, updates = deduplicate_and_merge(cards, db, embedder)
        assert len(new) == 2
        assert len(updates) == 0

    def test_intra_batch_dedup(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽", ["好家伙"]
        cards = [
            self._make_card("1", same_tmpl, description=same_desc, examples=same_ex),
            self._make_card("2", same_tmpl, description=same_desc, examples=same_ex),
        ]
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "candidate")):
            new, updates = deduplicate_and_merge(cards, db, embedder)
        assert len(new) == 1
        assert len(updates) == 0

    def test_cross_db_dedup(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(10, 10, 10, 10)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "candidate")):
            new, updates = deduplicate_and_merge([new_card], db, embedder)
        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].id == "existing-1"
        assert updates[0].frequency.total == 11


class TestJudgeDuplicateTopN:
    def _make_card(self, cid: str, template: str, description: str = "默认描述") -> PatternCard:
        return PatternCard(
            id=cid, description=description,
            template=template, examples=["示例"],
            frequency=FrequencyProfile(1, 1, 1, 1),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_finds_duplicate_in_candidates(self):
        resp = '{"duplicate_of": 1, "keep_description": "candidate", "reason": "same"}'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, keep = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]好家伙[B]"), self._make_card("c2", "[A]绝了")],
            )
        assert idx == 0
        assert keep == "candidate"

    def test_no_duplicate_returns_none(self):
        resp = '{"duplicate_of": 0, "keep_description": "current", "reason": "all different"}'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, keep = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了")],
            )
        assert idx is None

    def test_parse_error_returns_none(self):
        with patch("brain.extract.refiner._call_llm_streaming", return_value=("bad json", 5, 5)):
            idx, keep = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了")],
            )
        assert idx is None
        assert keep == "current"

    def test_handles_markdown_code_fence(self):
        resp = '```json\n{"duplicate_of": 2, "keep_description": "current", "reason": "same"}\n```'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, keep = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了"), self._make_card("c2", "[A]好家伙[B]")],
            )
        assert idx == 1
        assert keep == "current"

    def test_renders_candidates_into_prompt(self):
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=('{"duplicate_of": 0, "keep_description": "current", "reason": "different"}', 10, 20),
        ) as mock_call:
            _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了")],
            )

        rendered_prompt = mock_call.call_args.args[0]
        assert "--- 候选 1 (c1) ---" in rendered_prompt
        assert "{{ candidates_block }}" not in rendered_prompt

    def test_empty_candidates_returns_none(self):
        idx, keep = _judge_duplicate_topn(
            self._make_card("new", "[A]好家伙"),
            [],
        )
        assert idx is None


class TestPromptLoader:
    def test_loads_extract_prompts(self):
        assert "JSON 数组" in load_prompt("extract_patterns.txt")
        assert "duplicate_of" in load_prompt("extract_dedup_judge.txt")
        assert "回复评论" in load_prompt("extract_patterns.txt")


@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set — skipping integration test",
)
class TestExtractIntegration:
    def test_real_extraction(self):
        """Smoke test: send real comment pairs to LLM and check output structure."""
        comments = [
            "上文评论：这视频也太离谱了\n回复评论：这也行...好家伙...",
            "上文评论：又来这一套\n回复评论：又来了...好家伙...",
            "上文评论：这也能圆回来？\n回复评论：绝了...好家伙...",
            "上文评论：你觉得靠谱吗\n回复评论：建议下次不要建议了",
            "上文评论：这操作太草了\n回复评论：太真实了",
        ]
        cards, _ = extract_from_chunk(comments)
        assert isinstance(cards, list)
        if cards:
            card = cards[0]
            assert card.template
            assert len(card.examples) > 0


from brain.extract.refiner import _dedup_intra_batch
from brain.extract.refiner import _dedup_against_db
from brain.store.pattern_db import PatternDB
from helpers import MockEmbedder


class TestIntraBatchDedup:
    def _make_card(self, cid: str, template: str, description: str = "默认描述",
                   examples: list[str] | None = None) -> PatternCard:
        return PatternCard(
            id=cid, description=description,
            template=template, examples=examples or ["示例"],
            frequency=FrequencyProfile(1, 1, 1, 1),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_single_card_passes_through(self):
        embedder = MockEmbedder()
        card = self._make_card("1", "[A]模式")
        result = _dedup_intra_batch([card], embedder, top_n=3)
        assert len(result) == 1
        assert result[0].id == "1"

    def test_different_cards_both_kept(self):
        embedder = MockEmbedder()
        card_a = self._make_card("1", "[A]完全不同A", description="descA", examples=["exA"])
        card_b = self._make_card("2", "[A]完全不同B", description="descB", examples=["exB"])
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, "current")):
            result = _dedup_intra_batch([card_a, card_b], embedder, top_n=3)
        assert len(result) == 2

    def test_duplicate_cards_merged(self):
        embedder = MockEmbedder()
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description=same_desc, examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "current")):
            result = _dedup_intra_batch([card_a, card_b], embedder, top_n=3)
        assert len(result) == 1
        assert result[0].id == "1"
        assert result[0].frequency.total == 2

    def test_description_updated_when_keep_current(self):
        embedder = MockEmbedder()
        same_tmpl, same_ex = "[A]好家伙", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description="旧描述", examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description="旧描述", examples=same_ex)
        card_b.description = "更好的描述"
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "current")):
            result = _dedup_intra_batch([card_a, card_b], embedder, top_n=3)
        assert len(result) == 1
        assert result[0].description == "更好的描述"

    def test_non_duplicate_keeps_both(self):
        embedder = MockEmbedder()
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description=same_desc, examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, "current")):
            result = _dedup_intra_batch([card_a, card_b], embedder, top_n=3)
        assert len(result) == 2


class TestCrossDbDedup:
    def _make_card(self, cid: str, template: str, description: str = "默认描述",
                   examples: list[str] | None = None) -> PatternCard:
        return PatternCard(
            id=cid, description=description,
            template=template, examples=examples or ["示例"],
            frequency=FrequencyProfile(1, 1, 1, 1),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_empty_db_returns_all_as_new(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        cards = [self._make_card("1", "[A]模式A"), self._make_card("2", "[A]模式B")]
        new, updates = _dedup_against_db(cards, db, top_n=3)
        assert len(new) == 2
        assert len(updates) == 0

    def test_similar_to_existing_triggers_llm_and_updates(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(5, 5, 5, 5)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "candidate")):
            new, updates = _dedup_against_db([new_card], db, top_n=3)

        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].id == "existing-1"
        assert updates[0].frequency.total == 6

    def test_different_from_existing_added_as_new(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        existing = self._make_card("existing-1", "[A]完全不同A", description="descA", examples=["exA"])
        db.save([existing])

        new_card = self._make_card("new-1", "[A]完全不同B", description="descB", examples=["exB"])
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, "current")):
            new, updates = _dedup_against_db([new_card], db, top_n=3)

        assert len(new) == 1
        assert len(updates) == 0

    def test_description_updated_when_keep_current(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description="旧描述", examples=same_ex)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description="更好的新描述", examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "current")):
            new, updates = _dedup_against_db([new_card], db, top_n=3)

        assert len(updates) == 1
        assert updates[0].description == "更好的新描述"

    def test_two_new_cards_merge_into_same_existing(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(5, 5, 5, 5)
        db.save([existing])

        card_x = self._make_card("x", same_tmpl, description=same_desc, examples=same_ex)
        card_y = self._make_card("y", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, "candidate")):
            new, updates = _dedup_against_db([card_x, card_y], db, top_n=3)

        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].frequency.total == 7
