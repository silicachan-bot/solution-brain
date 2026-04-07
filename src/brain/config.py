from __future__ import annotations

import os
from pathlib import Path

# Paths
SOLUTION_BRAIN_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = SOLUTION_BRAIN_DIR / "data"
STATE_FILE = DATA_DIR / "state.json"
CHROMA_DIR = DATA_DIR / "chromadb"

BILIBILI_DB_PATH = Path(
    os.environ.get(
        "BILIBILI_DB_PATH",
        str(SOLUTION_BRAIN_DIR.parent / "raw-data-pipeline" / "bilibili" / "_storage" / "database" / "bilibili.db"),
    )
)

# LLM API (OpenAI-compatible)
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Embedding API (Qwen3-Embedding-4B, OpenAI-compatible)
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "https://api.deepinfra.com/v1/openai")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBED_DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "2048"))

# Extraction
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "50"))
MIN_COMMENT_LENGTH = int(os.environ.get("MIN_COMMENT_LENGTH", "5"))

# Retrieval
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "8"))
SIMILARITY_WEIGHT = 0.6
FRESHNESS_WEIGHT = 0.4
