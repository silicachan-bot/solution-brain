from __future__ import annotations

import pytest

from helpers import MockEmbedder


@pytest.fixture(autouse=True)
def patch_embed_dimensions(monkeypatch):
    monkeypatch.setattr("brain.config.EMBED_DIMENSIONS", MockEmbedder.DIM)
    monkeypatch.setattr("brain.store.pattern_db.EMBED_DIMENSIONS", MockEmbedder.DIM)


@pytest.fixture
def mock_embedder():
    return MockEmbedder()
