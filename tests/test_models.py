from datetime import datetime
from brain.models import PatternCard, FrequencyProfile, CleanedComment


class TestFrequencyProfile:
    def test_freshness_zero_total(self):
        fp = FrequencyProfile(recent=0, medium=0, long_term=0, total=0)
        assert fp.freshness == 0.0

    def test_freshness_trending_up(self):
        fp = FrequencyProfile(recent=80, medium=120, long_term=150, total=150)
        assert fp.freshness > 0.7

    def test_freshness_dead_pattern(self):
        fp = FrequencyProfile(recent=0, medium=2, long_term=150, total=300)
        assert fp.freshness < 0.2

    def test_freshness_range(self):
        for r, m, l, t in [(0, 0, 0, 0), (100, 100, 100, 100), (1, 50, 200, 500)]:
            fp = FrequencyProfile(recent=r, medium=m, long_term=l, total=t)
            assert 0.0 <= fp.freshness <= 1.0


class TestPatternCard:
    def test_create_pattern_card(self):
        fp = FrequencyProfile(recent=10, medium=20, long_term=50, total=80)
        card = PatternCard(
            id="pat-001",
            title="test pattern",
            description="a test",
            template="[A] test",
            examples=["hello test", "world test"],
            frequency=fp,
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        assert card.id == "pat-001"
        assert card.title == "test pattern"
        assert len(card.examples) == 2
        assert card.frequency.freshness > 0

    def test_to_dict_roundtrip(self):
        fp = FrequencyProfile(recent=5, medium=10, long_term=30, total=45)
        card = PatternCard(
            id="pat-002",
            title="roundtrip",
            description="test roundtrip",
            template="[A]...[B]",
            examples=["x...y"],
            frequency=fp,
            source="test",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        d = card.to_dict()
        restored = PatternCard.from_dict(d)
        assert restored.id == card.id
        assert restored.title == card.title
        assert restored.frequency.recent == card.frequency.recent


class TestCleanedComment:
    def test_create(self):
        c = CleanedComment(
            rpid=123,
            bvid="BV1test",
            uid=456,
            message="hello world",
            ctime=1700000000,
        )
        assert c.rpid == 123
        assert c.message == "hello world"
