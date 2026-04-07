import json
from unittest.mock import patch, MagicMock
from datetime import datetime

from brain.models import CleanedComment, PatternCard, FrequencyProfile
from brain.extract.chunker import chunk_comments
from brain.extract.refiner import extract_from_chunk, deduplicate_and_merge


def _make_comment(rpid: int, message: str) -> CleanedComment:
    return CleanedComment(rpid=rpid, bvid="BV1test", uid=1, message=message, ctime=1700000000)


class TestChunker:
    def test_basic_chunking(self):
        comments = [_make_comment(i, f"comment number {i}") for i in range(120)]
        chunks = chunk_comments(comments, chunk_size=50)
        assert len(chunks) == 3
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50
        assert len(chunks[2]) == 20

    def test_small_input(self):
        comments = [_make_comment(i, f"comment {i}") for i in range(10)]
        chunks = chunk_comments(comments, chunk_size=50)
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_empty_input(self):
        chunks = chunk_comments([], chunk_size=50)
        assert chunks == []

    def test_returns_message_strings(self):
        comments = [_make_comment(1, "hello world")]
        chunks = chunk_comments(comments, chunk_size=50)
        assert chunks == [["hello world"]]


class TestExtractFromChunk:
    def test_parses_llm_response(self):
        mock_patterns = [
            {
                "title": "好家伙式吐槽",
                "template": "[A]...好家伙...",
                "examples": ["这也行...好家伙...", "又来了...好家伙..."],
                "description": "表达无奈和吐槽",
            }
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_patterns, ensure_ascii=False)

        with patch("brain.extract.refiner._call_llm", return_value=mock_response):
            results = extract_from_chunk(["comment1", "comment2"])

        assert len(results) == 1
        assert results[0].title == "好家伙式吐槽"
        assert results[0].template == "[A]...好家伙..."

    def test_handles_empty_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[]"

        with patch("brain.extract.refiner._call_llm", return_value=mock_response):
            results = extract_from_chunk(["comment1", "comment2"])

        assert results == []


class TestDeduplicateAndMerge:
    def _make_card(self, id: str, title: str, recent: int = 5) -> PatternCard:
        return PatternCard(
            id=id,
            title=title,
            description=f"desc for {title}",
            template=f"[A] {title}",
            examples=[f"example of {title}"],
            frequency=FrequencyProfile(recent=recent, medium=recent, long_term=recent, total=recent),
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

    def test_no_duplicates(self):
        cards = [self._make_card("1", "pattern A"), self._make_card("2", "pattern B")]
        new, updates = deduplicate_and_merge(cards, existing=[])
        assert len(new) == 2
        assert len(updates) == 0

    def test_merges_similar_new_cards(self):
        cards = [
            self._make_card("1", "好家伙式吐槽", recent=3),
            self._make_card("2", "好家伙式吐槽", recent=5),
        ]
        new, updates = deduplicate_and_merge(cards, existing=[])
        assert len(new) == 1
        assert new[0].frequency.recent == 8

    def test_updates_existing_pattern(self):
        existing = [self._make_card("existing-1", "好家伙式吐槽", recent=10)]
        new_cards = [self._make_card("new-1", "好家伙式吐槽", recent=3)]
        new, updates = deduplicate_and_merge(new_cards, existing=existing)
        assert len(new) == 0
        assert len(updates) == 1
        assert updates[0].id == "existing-1"
        assert updates[0].frequency.recent == 13
