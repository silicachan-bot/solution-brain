from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "scripts" / "streamlit_patterns.py"
    sys.argv = ["streamlit", "run", str(target), *sys.argv[1:]]
    raise SystemExit(stcli.main())
