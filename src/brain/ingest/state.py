from __future__ import annotations

import json
from pathlib import Path


class WatermarkState:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._data: dict = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def get_watermark(self, source: str) -> str | None:
        return self._data.get(source)

    def set_watermark(self, source: str, value: str) -> None:
        self._data[source] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
