import json
import os
from unittest.mock import patch
from datetime import datetime

import pytest

from brain.models import CleanedComment, CommentPair, FrequencyProfile, PatternCard, PatternOrigin
from brain.extract.chunker import build_comment_pairs, chunk_comment_pairs, chunk_comments, format_comment_pair
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

    def test_chunk_comment_pairs_keeps_pair_objects(self):
        pairs = [
            CommentPair(
                parent=_make_comment(i, f"parent {i}"),
                reply=_make_comment(i + 1000, f"reply {i}", root=i, parent=i),
            )
            for i in range(3)
        ]
        chunks = chunk_comment_pairs(pairs, chunk_size=2)
        assert len(chunks) == 2
        assert chunks[0][0].reply.message == "reply 0"
        assert chunks[1][0].reply.message == "reply 2"

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

    def test_attaches_origins_for_exact_matching_examples(self):
        mock_patterns = [
            {
                "template": "神TM[A]",
                "examples": ["神TM夜神月[笑哭]"],
                "description": "表达惊讶和吐槽",
            }
        ]
        pair = CommentPair(
            parent=_make_comment(1, "这个比喻太怪了"),
            reply=_make_comment(2, "神TM夜神月[笑哭]", root=1, parent=1),
        )
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=(json.dumps(mock_patterns, ensure_ascii=False), 10, 20),
        ):
            cards, _ = extract_from_chunk(
                ["上文评论：这个比喻太怪了\n回复评论：神TM夜神月[笑哭]"],
                comment_pairs=[pair],
                video_title="测试视频",
            )

        assert len(cards) == 1
        assert len(cards[0].origins) == 1
        assert cards[0].origins[0].bvid == "BV1test"
        assert cards[0].origins[0].video_title == "测试视频"
        assert cards[0].origins[0].parent_message == "这个比喻太怪了"

    def test_skips_origin_when_example_not_exact_reply(self):
        mock_patterns = [
            {
                "template": "神TM[A]",
                "examples": ["神TM夜神月"],
                "description": "表达惊讶和吐槽",
            }
        ]
        pair = CommentPair(
            parent=_make_comment(1, "这个比喻太怪了"),
            reply=_make_comment(2, "神TM夜神月[笑哭]", root=1, parent=1),
        )
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=(json.dumps(mock_patterns, ensure_ascii=False), 10, 20),
        ):
            cards, _ = extract_from_chunk(
                ["上文评论：这个比喻太怪了\n回复评论：神TM夜神月[笑哭]"],
                comment_pairs=[pair],
                video_title="测试视频",
            )

        assert len(cards) == 1
        assert cards[0].origins == []


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
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, 0)):
            new, updates, tokens = deduplicate_and_merge(cards, db, embedder)
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
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)):
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, updates, tokens = deduplicate_and_merge(cards, db, embedder)
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
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)):
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, updates, tokens = deduplicate_and_merge([new_card], db, embedder)
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
        resp = '{"duplicate_of": 1, "reason": "same"}'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, tokens = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]好家伙[B]"), self._make_card("c2", "[A]绝了")],
            )
        assert idx == 0
        assert tokens == 30

    def test_no_duplicate_returns_none(self):
        resp = '{"duplicate_of": 0, "reason": "all different"}'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, tokens = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了")],
            )
        assert idx is None
        assert tokens == 30

    def test_parse_error_returns_none(self):
        with patch("brain.extract.refiner._call_llm_streaming", return_value=("bad json", 5, 5)):
            idx, tokens = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了")],
            )
        assert idx is None
        assert tokens == 10

    def test_handles_markdown_code_fence(self):
        resp = '```json\n{"duplicate_of": 2, "reason": "same"}\n```'
        with patch("brain.extract.refiner._call_llm_streaming", return_value=(resp, 10, 20)):
            idx, tokens = _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了"), self._make_card("c2", "[A]好家伙[B]")],
            )
        assert idx == 1
        assert tokens == 30

    def test_renders_candidates_into_prompt(self):
        with patch(
            "brain.extract.refiner._call_llm_streaming",
            return_value=('{"duplicate_of": 0, "reason": "different"}', 10, 20),
        ) as mock_call:
            _judge_duplicate_topn(
                self._make_card("new", "[A]好家伙"),
                [self._make_card("c1", "[A]绝了")],
            )

        rendered_prompt = mock_call.call_args.args[0]
        assert "--- 候选 1 (c1) ---" in rendered_prompt
        assert "{{ candidates_block }}" not in rendered_prompt

    def test_empty_candidates_returns_none(self):
        idx, tokens = _judge_duplicate_topn(
            self._make_card("new", "[A]好家伙"),
            [],
        )
        assert idx is None
        assert tokens == 0


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


from brain.extract.refiner import _dedup_single_pass
from brain.store.pattern_db import PatternDB
from helpers import MockEmbedder


class TestDedupSinglePass:
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

    def _make_origin(self, example: str, suffix: str) -> PatternOrigin:
        return PatternOrigin(
            example=example,
            bvid=f"BV{suffix}",
            video_title=f"视频{suffix}",
            parent_message=f"上文{suffix}",
            reply_message=example,
        )

    def _empty_db(self, tmp_path):
        return PatternDB(tmp_path / "lance", embedder=MockEmbedder())

    class _ScenarioEmbedder:
        def __init__(self):
            self.vectors = {
                "A": [1.0, 0.0],
                "B": [1.0, 0.0],
                "C": [0.0, 1.0],
                "oldA 例句：x": [1.0, 0.0],
                "oldB 例句：x": [1.0, 0.0],
                "oldC 例句：x": [0.0, 1.0],
                "merged 例句：x": [0.0, 1.0],
            }

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [self.vectors[text] for text in texts]

    # ── 空库场景 ────────────────────────────────────────────────────────

    def test_single_card_empty_db(self, tmp_path):
        db = self._empty_db(tmp_path)
        card = self._make_card("1", "[A]模式")
        new, updates, _ = _dedup_single_pass([card], db, MockEmbedder(), top_n=3)
        assert len(new) == 1 and new[0].id == "1"
        assert len(updates) == 0

    def test_two_different_cards_empty_db(self, tmp_path):
        db = self._empty_db(tmp_path)
        card_a = self._make_card("1", "[A]完全不同A", examples=["exA"])
        card_b = self._make_card("2", "[A]完全不同B", examples=["exB"])
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, 0)):
            new, updates, _ = _dedup_single_pass([card_a, card_b], db, MockEmbedder(), top_n=3)
        assert len(new) == 2
        assert len(updates) == 0

    # ── 批次内去重（intra-batch）────────────────────────────────────────

    def test_intra_batch_duplicate_merged(self, tmp_path):
        db = self._empty_db(tmp_path)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description=same_desc, examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)):
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, updates, _ = _dedup_single_pass([card_a, card_b], db, MockEmbedder(), top_n=3)
        assert len(new) == 1
        assert new[0].id == "1"
        assert new[0].frequency.total == 2

    def test_intra_batch_non_duplicate_keeps_both(self, tmp_path):
        db = self._empty_db(tmp_path)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description=same_desc, examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, 0)):
            new, updates, _ = _dedup_single_pass(
                [card_a, card_b], db, MockEmbedder(), top_n=3, auto_merge_threshold=1.1
            )
        assert len(new) == 2

    def test_intra_batch_auto_merge_skips_llm(self, tmp_path):
        db = self._empty_db(tmp_path)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description=same_desc, examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn") as mock_judge:
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, _, _ = _dedup_single_pass(
                    [card_a, card_b], db, MockEmbedder(), top_n=3,
                    similarity_threshold=0.8, auto_merge_threshold=0.0,
                )
        mock_judge.assert_not_called()
        assert len(new) == 1

    def test_intra_batch_high_sim_calls_llm(self, tmp_path):
        db = self._empty_db(tmp_path)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        card_a = self._make_card("1", same_tmpl, description=same_desc, examples=same_ex)
        card_b = self._make_card("2", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)) as mock_judge:
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                _dedup_single_pass(
                    [card_a, card_b], db, MockEmbedder(), top_n=3,
                    similarity_threshold=0.8, auto_merge_threshold=1.1,
                )
        mock_judge.assert_called_once()

    def test_intra_batch_examples_merged(self, tmp_path):
        db = self._empty_db(tmp_path)
        card_a = self._make_card("1", "[A]模式", examples=["旧例句"])
        card_b = self._make_card("2", "[A]模式", examples=["新例句", "旧例句"])
        card_a.origins = [self._make_origin("旧例句", "old")]
        card_b.origins = [self._make_origin("新例句", "new"), self._make_origin("旧例句", "new-old")]
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)):
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, _, _ = _dedup_single_pass([card_a, card_b], db, MockEmbedder(), top_n=3)
        assert len(new) == 1
        assert new[0].examples == ["旧例句", "新例句"]
        assert new[0].origins[0].bvid == "BVold"

    def test_intra_batch_refreshes_vectors_after_merge(self, tmp_path):
        embedder = self._ScenarioEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        card_a = self._make_card("1", "A", description="oldA", examples=["x"])
        card_b = self._make_card("2", "B", description="oldB", examples=["x"])
        card_c = self._make_card("3", "C", description="oldC", examples=["x"])

        with patch("brain.extract.refiner._enrich_description", return_value=("merged", 0)):
            new, updates, _ = _dedup_single_pass(
                [card_a, card_b, card_c],
                db,
                embedder,
                top_n=3,
                similarity_threshold=0.8,
                auto_merge_threshold=0.9,
            )

        assert len(new) == 1
        assert new[0].id == "1"
        assert new[0].frequency.total == 3
        assert len(updates) == 0

    # ── 与 DB 比对场景（cross-DB）──────────────────────────────────────

    def test_matches_existing_db(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(5, 5, 5, 5)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)):
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, updates, _ = _dedup_single_pass([new_card], db, embedder, top_n=3)
        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].id == "existing-1"
        assert updates[0].frequency.total == 6

    def test_different_from_existing_added_as_new(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        existing = self._make_card("existing-1", "[A]完全不同A", examples=["exA"])
        db.save([existing])

        new_card = self._make_card("new-1", "[A]完全不同B", examples=["exB"])
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(None, 0)):
            new, updates, _ = _dedup_single_pass([new_card], db, embedder, top_n=3)
        assert len(new) == 1
        assert len(updates) == 0

    def test_two_new_cards_merge_into_same_existing(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(5, 5, 5, 5)
        db.save([existing])

        card_x = self._make_card("x", same_tmpl, description=same_desc, examples=same_ex)
        card_y = self._make_card("y", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)):
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, updates, _ = _dedup_single_pass([card_x, card_y], db, embedder, top_n=3)
        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].frequency.total == 7

    def test_db_description_enriched_on_merge(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        same_tmpl, same_ex = "[A]好家伙", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description="旧描述", examples=same_ex)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description="更好的新描述", examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)):
            with patch("brain.extract.refiner._enrich_description", return_value=("增补后描述", 0)) as mock_enrich:
                new, updates, _ = _dedup_single_pass([new_card], db, embedder, top_n=3)
        assert len(updates) == 1
        assert mock_enrich.called
        assert updates[0].description == "增补后描述"

    def test_db_examples_merged(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        existing = self._make_card("existing-1", "[A]好家伙", examples=["旧例句"])
        existing.origins = [self._make_origin("旧例句", "old")]
        db.save([existing])

        new_card = self._make_card("new-1", "[A]好家伙", examples=["新例句", "旧例句"])
        new_card.origins = [self._make_origin("新例句", "new"), self._make_origin("旧例句", "new-old")]
        with patch("brain.extract.refiner._judge_duplicate_topn", return_value=(0, 0)):
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, updates, _ = _dedup_single_pass([new_card], db, embedder, top_n=3)
        assert len(new) == 0
        assert updates[0].examples == ["旧例句", "新例句"]
        assert updates[0].origins[0].bvid == "BVold"

    def test_low_similarity_skips_llm(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        existing = self._make_card("existing-1", "[A]完全不同A", examples=["exA"])
        db.save([existing])

        new_card = self._make_card("new-1", "[A]完全不同B", examples=["exB"])
        with patch("brain.extract.refiner._judge_duplicate_topn") as mock_judge:
            new, updates, _ = _dedup_single_pass(
                [new_card], db, embedder, top_n=3, similarity_threshold=0.8
            )
        assert len(new) == 1
        assert len(updates) == 0
        mock_judge.assert_not_called()

    def test_db_auto_merge_skips_llm(self, tmp_path):
        embedder = MockEmbedder()
        db = PatternDB(tmp_path / "lance", embedder=embedder)
        same_tmpl, same_desc, same_ex = "[A]好家伙", "吐槽表达", ["好家伙"]
        existing = self._make_card("existing-1", same_tmpl, description=same_desc, examples=same_ex)
        existing.frequency = FrequencyProfile(5, 5, 5, 5)
        db.save([existing])

        new_card = self._make_card("new-1", same_tmpl, description=same_desc, examples=same_ex)
        with patch("brain.extract.refiner._judge_duplicate_topn") as mock_judge:
            with patch("brain.extract.refiner._enrich_description", return_value=("增补描述", 0)):
                new, updates, _ = _dedup_single_pass(
                    [new_card], db, embedder, top_n=3,
                    similarity_threshold=0.8, auto_merge_threshold=0.0,
                )
        mock_judge.assert_not_called()
        assert len(new) == 0
        assert updates[0].id == "existing-1"
        assert updates[0].frequency.total == 6
