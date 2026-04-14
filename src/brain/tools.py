from __future__ import annotations

from brain.compose.tools import get_tool_definition, handle_inspect_pattern
from brain.memory.soul import Soul
from brain.memory.tools import get_memory_tools, handle_memory_tool
from brain.store.pattern_db import PatternDB


def get_tools(soul: Soul, db: PatternDB | None = None) -> list[dict]:
    """返回当前可用的所有 tool 定义。"""
    tools = get_memory_tools()          # soul：update_self_memory
    if db is not None:
        tools.append(get_tool_definition())  # patterns：inspect_pattern
    return tools


def dispatch(name: str, args: dict, soul: Soul, db: PatternDB | None = None) -> str:
    """将 tool call 路由到对应处理函数。"""
    if name == "update_self_memory":
        return handle_memory_tool(name, args, soul)
    if name == "inspect_pattern":
        if db is None:
            return "语言模式库当前不可用。"
        return handle_inspect_pattern(db, args.get("pattern_id", ""))
    return f"未知工具：{name}"
