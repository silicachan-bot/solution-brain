from __future__ import annotations

from brain.store.pattern_db import PatternDB


def get_tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "inspect_pattern",
            "description": "查看某个语言模式的详细描述和使用示例。使用任何模式前必须先查看。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_id": {
                        "type": "string",
                        "description": "模式 ID，如 pat-a3f2c1",
                    }
                },
                "required": ["pattern_id"],
            },
        },
    }


def handle_inspect_pattern(db: PatternDB, pattern_id: str) -> str:
    card = db.get(pattern_id)
    if card is None:
        return f"找不到 ID 为 {pattern_id} 的语言模式。"

    examples_text = "\n".join(f"  - {ex}" for ex in card.examples)
    return (
        f"模板：{card.template}\n"
        f"描述：{card.description}\n"
        f"例句：\n{examples_text}"
    )
