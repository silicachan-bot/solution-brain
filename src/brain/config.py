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

# ── 环境变量（写在 .env，因机器/账号/实验而异） ───────────────────────────────────
#
#   这里的值来自 .env 文件或系统环境变量，不要在代码里硬编码。
#   详见 .env.example。

# Bilibili SQLite 数据库路径（默认指向 monorepo 相对路径）
BILIBILI_DB_PATH = Path(
    os.environ.get(
        "BILIBILI_DB_PATH",
        str(MAIN_REPO_DIR.parent / "raw-data-pipeline" / "bilibili" / "_storage" / "database" / "bilibili.db"),
    )
)

# 提取模型（LLM，OpenAI-compatible）
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# 嵌入模型（OpenAI-compatible）
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "https://api.siliconflow.cn/v1")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
# 向量维度由模型决定，切换模型时需同步修改（也可覆盖到 .env）
EMBED_DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "2048"))

# ── 算法参数（直接在此修改，不需要写 .env） ──────────────────────────────────────
#
#   这些是程序行为的一部分，不含敏感信息，可随代码提交。
#   如需在不同环境快速切换，也可在 .env 中覆盖同名变量。

# LLM 调用超时秒数；网络较慢时可适当调大
LLM_TIMEOUT_SECONDS: float = 60

# 评论最短字数；过短的评论（纯表情、回复符号等）直接过滤
MIN_COMMENT_LENGTH: int = 5

# 每块包含的评论对数量；越大单次 LLM 调用的上下文越丰富，token 消耗也越高
CHUNK_SIZE: int = 100

# 每个视频最多处理的块数（0 = 全部）；用于控制高评论量视频的处理深度
MAX_CHUNKS_PER_VIDEO: int = 3

# retrieve_patterns() 默认返回的模式数量上限
RETRIEVAL_TOP_K: int = 8

# 综合检索得分 = 向量相似度 × SIMILARITY_WEIGHT + freshness × FRESHNESS_WEIGHT
SIMILARITY_WEIGHT: float = 0.6
FRESHNESS_WEIGHT: float = 0.4

# 去重双路向量检索各取 top-N 候选，再进入 LLM 判重或自动合并
DEDUP_TOP_N: int = 3

# 相似度低于此阈值的候选直接忽略，不进入后续判重
DEDUP_SIMILARITY_THRESHOLD: float = 0.8

# 相似度高于此阈值时直接自动合并，跳过 LLM 调用
DEDUP_AUTO_MERGE_THRESHOLD: float = 0.9
