from __future__ import annotations

import hashlib
import random as _random


class MockEmbedder:
    """确定性 mock embedder：相同文本 -> 相同单位向量（cos=1.0）；不同文本 -> 近似正交。"""
    DIM = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
            rng = _random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self.DIM)]
            norm = sum(x * x for x in vec) ** 0.5
            result.append([x / norm for x in vec])
        return result
