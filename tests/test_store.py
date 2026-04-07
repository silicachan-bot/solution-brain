from datetime import datetime
from unittest.mock import patch, MagicMock

from brain.models import PatternCard, FrequencyProfile
from brain.store.pattern_db import PatternDB
from brain.store.retriever import retrieve_patterns
from brain.store.embedding import QwenEmbeddingFunction


def _make_card(id: str, title: str, description: str = "test desc") -> PatternCard:
    return PatternCard(
        id=id,
        title=title,
        description=description,
        template=f"[A] {title}",
        examples=[f"example of {title}"],
        frequency=FrequencyProfile(recent=5, medium=10, long_term=30, total=45),
        source="test",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


class TestPatternDB:
    def test_save_and_get(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        card = _make_card("pat-001", "test pattern")
        db.save([card])
        retrieved = db.get("pat-001")
        assert retrieved is not None
        assert retrieved.title == "test pattern"
        assert retrieved.frequency.recent == 5

    def test_list_all(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        db.save([_make_card("1", "alpha"), _make_card("2", "beta")])
        all_cards = db.list_all()
        assert len(all_cards) == 2

    def test_update(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        card = _make_card("pat-001", "test pattern")
        db.save([card])
        card.frequency.recent = 99
        card.examples.append("new example")
        db.update([card])
        retrieved = db.get("pat-001")
        assert retrieved.frequency.recent == 99
        assert "new example" in retrieved.examples

    def test_save_empty_list(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        db.save([])
        assert db.list_all() == []


class TestRetriever:
    def test_retrieve_returns_cards(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        db.save([
            _make_card("1", "吐槽表达", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "撒娇语气", "用叠字和语气词表达撒娇"),
            _make_card("3", "反问句式", "用反问表达不满"),
        ])
        results = retrieve_patterns(db, "这也太离谱了吧", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(c, PatternCard) for c in results)

    def test_retrieve_empty_db(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        results = retrieve_patterns(db, "test query", top_k=5)
        assert results == []


class TestQwenEmbedding:
    def test_call_returns_embeddings(self):
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 2048),
            MagicMock(embedding=[0.2] * 2048),
        ]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("brain.store.embedding.OpenAI", return_value=mock_client):
            ef = QwenEmbeddingFunction()
            result = ef(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 2048
