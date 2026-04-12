from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _resolve_main_repo_dir(worktree_dir: Path) -> Path:
    """在 .claude/worktrees/* 下运行时，回退到主仓库根目录。"""
    if worktree_dir.parent.name == "worktrees" and worktree_dir.parent.parent.name == ".claude":
        return worktree_dir.parent.parent.parent
    return worktree_dir


# Paths
SOLUTION_BRAIN_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_REPO_DIR = _resolve_main_repo_dir(SOLUTION_BRAIN_DIR)
load_dotenv(SOLUTION_BRAIN_DIR / ".env")

DATA_DIR = SOLUTION_BRAIN_DIR / "data"
STATE_FILE = DATA_DIR / "state.json"
LANCEDB_DIR = DATA_DIR / "lancedb"

BILIBILI_DB_PATH = Path(
    os.environ.get(
        "BILIBILI_DB_PATH",
        str(MAIN_REPO_DIR.parent / "raw-data-pipeline" / "bilibili" / "_storage" / "database" / "bilibili.db"),
    )
)

# LLM API (OpenAI-compatible)
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))

# Embedding API (Qwen3-Embedding-4B via SiliconFlow, OpenAI-compatible)
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "https://api.siliconflow.cn/v1")
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
DEDUP_TOP_N = int(os.environ.get("DEDUP_TOP_N", "3"))
