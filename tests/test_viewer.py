from datetime import datetime

from brain.models import FrequencyProfile, PatternCard, PatternOrigin
from brain.viewer import filter_patterns, format_pattern_summary, group_origins_by_example, sort_patterns


def _make_card(
    id: str,
    *,
    description: str,
    template: str,
    examples: list[str],
    updated_day: int,
    recent: int,
    medium: int,
    total: int,
) -> PatternCard:
    return PatternCard(
        id=id,
        description=description,
        template=template,
        examples=examples,
        frequency=FrequencyProfile(
            recent=recent,
            medium=medium,
            long_term=total,
            total=total,
        ),
        source="test",
        created_at=datetime(2026, 4, 1),
        updated_at=datetime(2026, 4, updated_day),
    )


class TestFilterPatterns:
    def test_filter_patterns_matches_template(self):
        cards = [
            _make_card(
                "1",
                description="对离谱情况吐槽",
                template="[A]...好家伙...",
                examples=["这也行...好家伙..."],
                updated_day=1,
                recent=4,
                medium=8,
                total=10,
            ),
            _make_card(
                "2",
                description="用反问表达不满",
                template="这也叫[A]？",
                examples=["这也叫解释？"],
                updated_day=2,
                recent=2,
                medium=4,
                total=6,
            ),
        ]

        result = filter_patterns(cards, "好家伙")

        assert [card.id for card in result] == ["1"]

    def test_filter_patterns_matches_examples_and_description(self):
        cards = [
            _make_card(
                "1",
                description="对离谱情况吐槽",
                template="[A]...好家伙...",
                examples=["这也行...好家伙..."],
                updated_day=1,
                recent=4,
                medium=8,
                total=10,
            ),
            _make_card(
                "2",
                description="用叠字表达亲近",
                template="[A]啦~",
                examples=["拜托拜托啦~"],
                updated_day=2,
                recent=2,
                medium=4,
                total=6,
            ),
        ]

        by_description = filter_patterns(cards, "亲近")
        by_example = filter_patterns(cards, "拜托")

        assert [card.id for card in by_description] == ["2"]
        assert [card.id for card in by_example] == ["2"]

    def test_filter_patterns_blank_query_returns_all(self):
        cards = [
            _make_card(
                "1",
                description="对离谱情况吐槽",
                template="[A]...好家伙...",
                examples=["这也行...好家伙..."],
                updated_day=1,
                recent=4,
                medium=8,
                total=10,
            )
        ]

        result = filter_patterns(cards, "   ")

        assert [card.id for card in result] == ["1"]


class TestSortPatterns:
    def test_sort_patterns_by_updated_at_desc(self):
        cards = [
            _make_card(
                "1",
                description="a",
                template="[A]",
                examples=["a"],
                updated_day=1,
                recent=1,
                medium=2,
                total=10,
            ),
            _make_card(
                "2",
                description="b",
                template="[B]",
                examples=["b"],
                updated_day=7,
                recent=1,
                medium=2,
                total=10,
            ),
        ]

        result = sort_patterns(cards, "updated_at")

        assert [card.id for card in result] == ["2", "1"]

    def test_sort_patterns_by_freshness_desc(self):
        cards = [
            _make_card(
                "1",
                description="a",
                template="[A]",
                examples=["a"],
                updated_day=1,
                recent=1,
                medium=2,
                total=10,
            ),
            _make_card(
                "2",
                description="b",
                template="[B]",
                examples=["b"],
                updated_day=1,
                recent=8,
                medium=9,
                total=10,
            ),
        ]

        result = sort_patterns(cards, "freshness")

        assert [card.id for card in result] == ["2", "1"]

    def test_sort_patterns_by_template_asc(self):
        cards = [
            _make_card(
                "1",
                description="a",
                template="[A]吐槽",
                examples=["a"],
                updated_day=1,
                recent=1,
                medium=2,
                total=10,
            ),
            _make_card(
                "2",
                description="b",
                template="[A]反问",
                examples=["b"],
                updated_day=1,
                recent=8,
                medium=9,
                total=10,
            ),
        ]

        result = sort_patterns(cards, "template")

        assert [card.template for card in result] == ["[A]反问", "[A]吐槽"]


class TestFormatPatternSummary:
    def test_format_pattern_summary_contains_key_fields(self):
        card = _make_card(
            "1",
            description="对离谱情况吐槽",
            template="[A]...好家伙...",
            examples=["这也行...好家伙..."],
            updated_day=3,
            recent=4,
            medium=8,
            total=10,
        )

        summary = format_pattern_summary(card)

        assert "[A]...好家伙..." in summary
        assert "freshness=" in summary
        assert "total=10" in summary


class TestGroupOriginsByExample:
    def test_groups_origins_for_each_example(self):
        card = _make_card(
            "1",
            description="对离谱情况吐槽",
            template="[A]...好家伙...",
            examples=["这也行...好家伙...", "又来了...好家伙..."],
            updated_day=3,
            recent=4,
            medium=8,
            total=10,
        )
        card.origins = [
            PatternOrigin(
                example="这也行...好家伙...",
                bvid="BV1a",
                video_title="视频A",
                parent_message="上文A",
                reply_message="这也行...好家伙...",
            ),
            PatternOrigin(
                example="又来了...好家伙...",
                bvid="BV1b",
                video_title="视频B",
                parent_message="上文B",
                reply_message="又来了...好家伙...",
            ),
        ]

        grouped = group_origins_by_example(card)

        assert grouped["这也行...好家伙..."][0].bvid == "BV1a"
        assert grouped["又来了...好家伙..."][0].bvid == "BV1b"
