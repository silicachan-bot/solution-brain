import sqlite3
from pathlib import Path

from brain.ingest.cleaner import clean_comments
from brain.ingest.reader import BilibiliReader
from brain.ingest.state import WatermarkState
from brain.models import CleanedComment


def _create_test_db(path: Path):
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE videos (
            bvid TEXT PRIMARY KEY,
            title TEXT,
            crawl_status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE comments (
            rpid INTEGER PRIMARY KEY,
            bvid TEXT NOT NULL,
            uid INTEGER,
            uname TEXT,
            message TEXT,
            like INTEGER,
            ctime INTEGER,
            root INTEGER DEFAULT 0,
            parent INTEGER DEFAULT 0,
            rcount INTEGER DEFAULT 0
        )
    """)
    conn.execute("INSERT INTO videos VALUES ('BV1test', 'Test Video', 'completed')")
    conn.execute("INSERT INTO videos VALUES ('BV2test', 'Video 2', 'completed')")

    conn.execute(
        "INSERT INTO comments VALUES (1, 'BV1test', 101, 'alice', 'root comment', 10, 1700000001, 0, 0, 0)"
    )
    conn.execute(
        "INSERT INTO comments VALUES (2, 'BV1test', 102, 'bob', 'reply comment', 5, 1700000002, 1, 1, 0)"
    )
    conn.execute(
        "INSERT INTO comments VALUES (3, 'BV1test', 103, 'carol', 'another root comment', 3, 1700000003, 0, 0, 0)"
    )
    conn.execute(
        "INSERT INTO comments VALUES (4, 'BV2test', 201, 'dave', 'video 2 comment 1', 5, 1700000010, 0, 0, 0)"
    )
    conn.execute(
        "INSERT INTO comments VALUES (5, 'BV2test', 202, 'erin', 'video 2 comment 2', 5, 1700000011, 0, 0, 0)"
    )

    conn.commit()
    conn.close()


class TestBilibiliReader:
    def test_list_videos(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)
        reader = BilibiliReader(db_path)
        videos = reader.list_videos()
        assert len(videos) == 2

    def test_read_comments_for_video(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)
        reader = BilibiliReader(db_path)
        comments = reader.read_comments("BV1test")
        assert len(comments) == 3
        assert all(isinstance(c, CleanedComment) for c in comments)
        assert comments[0].bvid == "BV1test"
        assert comments[0].uname == "alice"
        assert comments[1].root == 1
        assert comments[1].parent == 1

    def test_read_comments_empty_video(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)
        reader = BilibiliReader(db_path)
        comments = reader.read_comments("BV_nonexistent")
        assert comments == []


def _make_comment(
    rpid: int,
    message: str,
    uid: int = 1,
    uname: str = "user",
    root: int = 0,
    parent: int = 0,
) -> CleanedComment:
    return CleanedComment(
        rpid=rpid,
        bvid="BV1test",
        uid=uid,
        uname=uname,
        message=message,
        ctime=1700000000,
        root=root,
        parent=parent,
    )


class TestCleaner:
    def test_removes_short_comments(self):
        comments = [
            _make_comment(1, "hi"),
            _make_comment(2, "这是一条足够长的评论"),
        ]
        result = clean_comments(comments)
        assert len(result) == 1
        assert result[0].rpid == 2

    def test_removes_pure_emoji(self):
        comments = [
            _make_comment(1, "😂😂😂"),
            _make_comment(2, "这也太好笑了😂"),
        ]
        result = clean_comments(comments)
        assert len(result) == 1
        assert result[0].rpid == 2

    def test_deduplicates_exact_messages(self):
        comments = [
            _make_comment(1, "这是一条重复的评论", uid=1),
            _make_comment(2, "这是一条重复的评论", uid=2),
            _make_comment(3, "这是一条不同的评论", uid=3),
        ]
        result = clean_comments(comments)
        messages = [c.message for c in result]
        assert messages.count("这是一条重复的评论") == 1
        assert "这是一条不同的评论" in messages

    def test_empty_input(self):
        assert clean_comments([]) == []


class TestWatermarkState:
    def test_get_default_watermark(self, tmp_path):
        state = WatermarkState(tmp_path / "state.json")
        assert state.get_watermark("bilibili") is None

    def test_set_and_get_watermark(self, tmp_path):
        state = WatermarkState(tmp_path / "state.json")
        state.set_watermark("bilibili", "BV1test")
        assert state.get_watermark("bilibili") == "BV1test"

    def test_persistence(self, tmp_path):
        path = tmp_path / "state.json"
        state1 = WatermarkState(path)
        state1.set_watermark("bilibili", "BV1abc")
        state2 = WatermarkState(path)
        assert state2.get_watermark("bilibili") == "BV1abc"

    def test_multiple_sources(self, tmp_path):
        state = WatermarkState(tmp_path / "state.json")
        state.set_watermark("bilibili", "BV1test")
        state.set_watermark("other_source", "id-999")
        assert state.get_watermark("bilibili") == "BV1test"
        assert state.get_watermark("other_source") == "id-999"
