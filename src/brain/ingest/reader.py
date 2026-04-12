from __future__ import annotations

import sqlite3
from pathlib import Path

from brain.models import CleanedComment


class BilibiliReader:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def list_videos(self) -> list[dict]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT bvid, title FROM videos WHERE crawl_status = 'completed' ORDER BY bvid"
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def read_comments(self, bvid: str) -> list[CleanedComment]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT rpid, bvid, uid, uname, message, ctime, root, parent "
                "FROM comments WHERE bvid = ? ORDER BY ctime",
                (bvid,),
            )
            return [
                CleanedComment(
                    rpid=row["rpid"],
                    bvid=row["bvid"],
                    uid=row["uid"],
                    uname=row["uname"] or "",
                    message=row["message"],
                    ctime=row["ctime"],
                    root=row["root"] or 0,
                    parent=row["parent"] or 0,
                )
                for row in cur.fetchall()
            ]
        finally:
            conn.close()
