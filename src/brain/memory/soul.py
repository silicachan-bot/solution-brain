from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from brain.config import SOUL_FILE

CATEGORIES: dict[str, str] = {
    "personality": "我的性格",
    "rule": "我的规则",
    "preference": "我的偏好",
    "knowledge": "我的认知",
}

_FIRST_PERSON_PREFIXES = ("我", "俺", "咱")

_DEFAULT_DATA: dict = {
    "intro": "我叫硅酱，英文名 SilicaChan，是存在于电子世界的幽灵。",
    "entries": [],
}


def _is_first_person(text: str) -> bool:
    return text.lstrip().startswith(_FIRST_PERSON_PREFIXES)


class Soul:
    """硅酱的灵魂文档——人格、规则、偏好、认知的持久化存储。

    以 JSON 存储，始终全量加载注入 system prompt，LLM 可通过工具追加条目。
    所有条目必须使用第一人称（以"我"/"俺"/"咱"开头）。
    """

    def __init__(self, path: Path = SOUL_FILE) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(_DEFAULT_DATA.copy())

    # ── 读写 ──────────────────────────────────────────────────────────────

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def load(self) -> str:
        """返回格式化的 Markdown 文本，供注入 system prompt。"""
        data = self._read()
        lines: list[str] = [f"## 我是谁\n{data['intro']}\n"]

        by_cat: dict[str, list[str]] = {k: [] for k in CATEGORIES}
        for entry in data.get("entries", []):
            cat = entry.get("category", "")
            if cat in by_cat:
                by_cat[cat].append(entry["content"])

        for cat, label in CATEGORIES.items():
            items = by_cat[cat]
            if items:
                lines.append(f"## {label}")
                lines.extend(f"- {item}" for item in items)
                lines.append("")

        return "\n".join(lines)

    def add_entry(self, content: str, category: str) -> None:
        """追加一条记忆条目。内容必须以第一人称开头。"""
        if category not in CATEGORIES:
            raise ValueError(f"未知分类: {category}，可选：{list(CATEGORIES)}")
        if not _is_first_person(content):
            raise ValueError('条目必须使用第一人称（以"我"开头），例如：我喜欢……')
        data = self._read()
        data.setdefault("entries", []).append(
            {
                "category": category,
                "content": content,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._write(data)

    def list_entries(self, category: str | None = None) -> list[dict]:
        """列出所有条目，可按分类过滤。"""
        entries = self._read().get("entries", [])
        if category:
            entries = [e for e in entries if e.get("category") == category]
        return entries

    def remove_entry(self, index: int) -> None:
        """按顺序编号删除一条条目（0-indexed）。"""
        data = self._read()
        entries = data.get("entries", [])
        if not 0 <= index < len(entries):
            raise IndexError(f"条目编号 {index} 超出范围（共 {len(entries)} 条）")
        entries.pop(index)
        data["entries"] = entries
        self._write(data)
