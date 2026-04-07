from __future__ import annotations

from chromadb import EmbeddingFunction, Documents, Embeddings
from openai import OpenAI

from brain.config import EMBED_API_BASE, EMBED_API_KEY, EMBED_MODEL, EMBED_DIMENSIONS


class QwenEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        self._client = OpenAI(base_url=EMBED_API_BASE, api_key=EMBED_API_KEY)

    def __call__(self, input: Documents) -> Embeddings:
        response = self._client.embeddings.create(
            model=EMBED_MODEL,
            input=input,
            dimensions=EMBED_DIMENSIONS,
        )
        return [item.embedding for item in response.data]
