from __future__ import annotations

from openai import OpenAI

from brain.config import EMBED_API_BASE, EMBED_API_KEY, EMBED_MODEL, EMBED_DIMENSIONS


class QwenEmbedder:
    """封装 OpenAI 兼容的 embedding API，返回原始向量列表。"""

    def __init__(self):
        self._client = OpenAI(base_url=EMBED_API_BASE, api_key=EMBED_API_KEY)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            dimensions=EMBED_DIMENSIONS,
        )
        return [item.embedding for item in response.data]
