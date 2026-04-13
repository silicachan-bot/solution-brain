#!/usr/bin/env bash
# 清空数据库并对指定 bvid 重跑提取，用于迭代对比
# 用法：bash scripts/clear_and_eval.sh BV1xxx BV1yyy BV1zzz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

if [ "$#" -lt 1 ]; then
  echo "用法: bash scripts/clear_and_eval.sh <bvid> [bvid ...]"
  exit 1
fi

echo "=== 清空数据库 ==="
rm -rf "$ROOT/data/lancedb"
rm -f  "$ROOT/data/state.json"
rm -f  "$ROOT/data/llm_responses.log"
echo "data/ 运行产物已清空"

echo ""
echo "=== 运行提取 (bvids: $*) ==="
cd "$ROOT"
uv run python scripts/eval_extract_samples.py "$@"
