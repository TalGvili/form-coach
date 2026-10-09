"""Tests for app.db, on a temporary database file: nothing here needs a video."""

import dataclasses
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app import db as store
from pipeline.models import Rep, SessionResult, VideoInfo

VIDEO = VideoInfo(path="upload.mp4", fps=30.0, width=1920, height=1080, n_frames=300)
REP = Rep(1, 0, 30, 60, 85.0, 4.0, 170.0, 2.0, 1.0, 0.0, 0.0, 28, 30, 59, 40, 50)
UNMEASURED = dataclasses.replace(REP, index=2, start_frame=60, max_elbow_angle=math.nan)
MONDAY = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    connection = store.connect(tmp_path / "history.db")
    yield connection
    connection.close()


def save(db, reps=(REP, UNMEASURED), profile="tal", video_id="a" * 32, when=MONDAY) -> int:
    result = SessionResult(VIDEO, list(reps), [], [])
    return store.save_session(
        db,
        result,
        profile=profile,
        exercise="pushup",
        video_id=video_id,
        created_at=when,
        video_starts={1: 0.0, 2: 3.5},
    )


def test_a_session_comes_back_with_the_same_reps(db):
    session = store.get_session(db, save(db))
    assert (session.profile, session.fps, session.n_frames) == ("tal", 30.0, 300)
    assert (session.width, session.height) == (1920, 1080)
    assert session.video_starts == {1: 0.0, 2: 3.5}
    assert session.created_at == "2026-10-05T18:00:00+00:00"
    assert session.reps[0] == REP
    # NaN != NaN, so compare the unmeasured rep field by field
    back = session.reps[1]
    assert math.isnan(back.max_elbow_angle)
    assert dataclasses.replace(back, max_elbow_angle=0.0) == dataclasses.replace(
        UNMEASURED, max_elbow_angle=0.0
    )


def test_an_unknown_session_is_none(db):
    assert store.get_session(db, 99) is None


def test_each_profile_sees_only_its_own_sessions_oldest_first(db):
    later = datetime(2026, 10, 7, 18, 0, tzinfo=UTC)
    save(db, profile="tal", video_id="b" * 32, when=later)
    save(db, profile="tal", video_id="a" * 32, when=MONDAY)
    save(db, profile="dana", video_id="c" * 32)
    sessions = store.list_sessions(db, "tal")
    assert [s.video_id for s in sessions] == ["a" * 32, "b" * 32]
    assert all(len(s.reps) == 2 for s in sessions)
    assert store.list_profiles(db) == ["dana", "tal"]


def test_a_failed_save_leaves_nothing_behind(db):
    """Two reps with the same index break the reps table's key after the session row went
    in: the transaction must take that row back out."""
    with pytest.raises(sqlite3.IntegrityError):
        save(db, reps=(REP, REP))
    assert store.list_sessions(db, "tal") == []


def test_the_reps_table_has_a_column_for_every_rep_field(db):
    columns = [row["name"] for row in db.execute("PRAGMA table_info(reps)")]
    assert columns == ["session_id", *(f.name for f in dataclasses.fields(Rep)), "video_start_s"]


def test_history_survives_closing_the_database(tmp_path: Path):
    path = tmp_path / "history.db"
    first = store.connect(path)
    session_id = save(first)
    first.close()
    second = store.connect(path)  # creating the tables again must not wipe them
    assert store.get_session(second, session_id).reps[0] == REP
    second.close()
