from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class FrequencyProfile:
    recent: int = 0
    medium: int = 0
    long_term: int = 0
    total: int = 0

    @property
    def freshness(self) -> float:
        if self.total == 0:
            return 0.0
        recent_ratio = self.recent / self.total
        medium_ratio = self.medium / self.total
        return recent_ratio * 0.35 + medium_ratio * 0.65


@dataclass
class PatternCard:
    id: str
    description: str
    template: str
    examples: list[str]
    frequency: FrequencyProfile
    source: str
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "template": self.template,
            "examples": self.examples,
            "frequency": {
                "recent": self.frequency.recent,
                "medium": self.frequency.medium,
                "long_term": self.frequency.long_term,
                "total": self.frequency.total,
            },
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PatternCard:
        freq = d["frequency"]
        return cls(
            id=d["id"],
            description=d["description"],
            template=d["template"],
            examples=d["examples"],
            frequency=FrequencyProfile(
                recent=freq["recent"],
                medium=freq["medium"],
                long_term=freq["long_term"],
                total=freq["total"],
            ),
            source=d["source"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )

    def embed_text(self) -> str:
        """语义检索用的 embedding 文本（description + examples）。"""
        examples_text = " / ".join(self.examples)
        return f"{self.description} 例句：{examples_text}"


@dataclass
class CleanedComment:
    rpid: int
    bvid: str
    uid: int
    message: str
    ctime: int
