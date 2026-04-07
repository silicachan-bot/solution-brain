from datetime import datetime

from brain.models import PatternCard, FrequencyProfile
from brain.store.pattern_db import PatternDB
from brain.compose.menu import build_menu
from brain.compose.tools import get_tool_definition, handle_inspect_pattern
from brain.compose.assembler import assemble_system_prompt


def _make_card(id: str, title: str, template: str) -> PatternCard:
    return PatternCard(
        id=id,
        title=title,
        description=f"desc for {title}",
        template=template,
        examples=[f"example of {title}"],
        frequency=FrequencyProfile(recent=5, medium=10, long_term=30, total=45),
        source="test",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


class TestBuildMenu:
    def test_formats_pattern_list(self):
        cards = [
            _make_card("pat-001", "好家伙式吐槽", "[A]...好家伙..."),
            _make_card("pat-002", "叠字撒娇", "[A]嘛~[B]啦~"),
        ]
        menu = build_menu(cards)
        assert "[pat-001]" in menu
        assert "好家伙式吐槽" in menu
        assert "[A]...好家伙..." in menu
        assert "[pat-002]" in menu

    def test_empty_patterns(self):
        menu = build_menu([])
        assert "没有" in menu or "无" in menu or menu.strip() == ""


class TestToolDefinition:
    def test_tool_schema(self):
        tool = get_tool_definition()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "inspect_pattern"
        assert "pattern_id" in tool["function"]["parameters"]["properties"]


class TestHandleInspectPattern:
    def test_returns_pattern_details(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        card = _make_card("pat-001", "好家伙式吐槽", "[A]...好家伙...")
        db.save([card])
        result = handle_inspect_pattern(db, "pat-001")
        assert "好家伙式吐槽" in result
        assert "[A]...好家伙..." in result
        assert "desc for" in result

    def test_pattern_not_found(self, tmp_path):
        db = PatternDB(tmp_path / "chroma")
        result = handle_inspect_pattern(db, "nonexistent")
        assert "找不到" in result or "not found" in result.lower()


class TestAssembler:
    def test_assembles_with_patterns(self):
        cards = [
            _make_card("pat-001", "好家伙式吐槽", "[A]...好家伙..."),
            _make_card("pat-002", "叠字撒娇", "[A]嘛~[B]啦~"),
        ]
        prompt = assemble_system_prompt(cards)
        assert "好家伙式吐槽" in prompt
        assert "叠字撒娇" in prompt
        assert "inspect_pattern" in prompt or "查看" in prompt

    def test_assembles_without_patterns(self):
        prompt = assemble_system_prompt([])
        assert isinstance(prompt, str)
        assert len(prompt) > 0
