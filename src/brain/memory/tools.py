from __future__ import annotations

from brain.memory.soul import CATEGORIES, Soul


def get_memory_tools() -> list[dict]:
    """返回自我记忆相关的 tool 定义列表（OpenAI function calling 格式）。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "update_self_memory",
                "description": (
                    "将新的认知、规则或偏好写入你的灵魂记忆。"
                    "当用户对你的说话方式提出反馈、或你意识到需要记住某件事时调用。"
                    '内容必须以第一人称书写（以"我"开头），例如：我喜欢……、我不会……'
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entry": {
                            "type": "string",
                            "description": '要记录的内容，一句话描述，必须以"我"开头',
                        },
                        "category": {
                            "type": "string",
                            "enum": list(CATEGORIES.keys()),
                            "description": (
                                "记忆分类："
                                "personality=性格特点，"
                                "rule=行为规则，"
                                "preference=偏好，"
                                "knowledge=认知/观点"
                            ),
                        },
                    },
                    "required": ["entry", "category"],
                },
            },
        }
    ]


def handle_memory_tool(
    name: str, args: dict, soul: Soul
) -> str:
    """执行记忆工具调用，返回结果文本。"""
    if name == "update_self_memory":
        entry = args.get("entry", "").strip()
        category = args.get("category", "")
        if not entry:
            return "记录内容不能为空。"
        try:
            soul.add_entry(entry, category)
            label = CATEGORIES.get(category, category)
            return f"已记录到「{label}」：{entry}"
        except ValueError as e:
            return str(e)
    return f"未知工具：{name}"
