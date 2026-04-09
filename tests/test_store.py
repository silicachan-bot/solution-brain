from datetime import datetime
from unittest.mock import patch, MagicMock

from brain.models import PatternCard, FrequencyProfile
from brain.store.pattern_db import PatternDB
from brain.store.retriever import retrieve_patterns
from brain.store.embedding import QwenEmbedder

from helpers import MockEmbedder


def _make_card(
    id: str, template: str = "[A] test", description: str = "test desc",
) -> PatternCard:
    return PatternCard(
        id=id,
        description=description,
        template=template,
        examples=[f"example of {template}"],
        frequency=FrequencyProfile(recent=5, medium=10, long_term=30, total=45),
        source="test",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


class TestPatternDB:
    def test_save_and_get(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        card = _make_card("pat-001", "[A] test pattern")
        db.save([card])
        retrieved = db.get("pat-001")
        assert retrieved is not None
        assert retrieved.template == "[A] test pattern"
        assert retrieved.frequency.recent == 5

    def test_list_all(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([_make_card("1", "[A] alpha"), _make_card("2", "[A] beta")])
        all_cards = db.list_all()
        assert len(all_cards) == 2

    def test_update(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        card = _make_card("pat-001", "[A] test pattern")
        db.save([card])
        card.frequency.recent = 99
        card.examples.append("new example")
        db.update([card])
        retrieved = db.get("pat-001")
        assert retrieved.frequency.recent == 99
        assert "new example" in retrieved.examples

    def test_save_empty_list(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([])
        assert db.list_all() == []

    def test_count(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        assert db.count() == 0
        db.save([_make_card("1"), _make_card("2")])
        assert db.count() == 2

    def test_query_by_template(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([
            _make_card("1", "[A]太离谱了"),
            _make_card("2", "[A]嘛~[B]啦~"),
        ])
        results = db.query_by_template("[A]太离谱了", top_k=1)
        assert len(results) == 1
        card, sim = results[0]
        assert isinstance(sim, float)

    def test_query_by_semantic(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([
            _make_card("1", "[A]太离谱了", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "[A]嘛~[B]啦~", "用叠字和语气词表达撒娇"),
        ])
        results = db.query_by_semantic("对离谱事情表达无奈的吐槽", top_k=1)
        assert len(results) == 1

    def test_query_empty_db(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        assert db.query_by_template("test") == []
        assert db.query_by_semantic("test") == []


class TestRetriever:
    def test_retrieve_returns_cards(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([
            _make_card("1", "[A]太离谱了", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "[A]嘛~[B]啦~", "用叠字和语气词表达撒娇"),
            _make_card("3", "[A]难道不是[B]吗", "用反问表达不满"),
        ])
        results = retrieve_patterns(db, "这也太离谱了吧", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(c, PatternCard) for c in results)

    def test_retrieve_empty_db(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        results = retrieve_patterns(db, "test query", top_k=5)
        assert results == []

    def test_retrieve_respects_top_k(self, tmp_path):
        db = PatternDB(tmp_path / "lance", embedder=MockEmbedder())
        db.save([
            _make_card("1", "[A]太离谱了", "对离谱事情表达无奈的吐槽"),
            _make_card("2", "[A]嘛~[B]啦~", "用叠字和语气词表达撒娇"),
            _make_card("3", "[A]难道不是[B]吗", "用反问表达不满"),
        ])
        results = retrieve_patterns(db, "这也太离谱了吧", top_k=1)
        assert len(results) == 1


class TestQwenEmbedder:
    def test_embed_returns_vectors(self):
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 2048),
            MagicMock(embedding=[0.2] * 2048),
        ]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        with patch("brain.store.embedding.OpenAI", return_value=mock_client):
            embedder = QwenEmbedder()
            result = embedder.embed(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 2048

    def test_embed_empty_list(self):
        embedder = QwenEmbedder.__new__(QwenEmbedder)
        assert embedder.embed([]) == []
