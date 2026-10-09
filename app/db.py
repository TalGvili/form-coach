"""SQLite session history: schema, save_session, and read queries returning dataclasses.

Measurements are stored, never verdicts. A rep row holds the Rep's numbers, not "this rep was
shallow": the faults are recomputed from the current config whenever a session is read, so
old sessions are judged by today's thresholds and history stays comparable after a tuning
change.

Reps are stored exactly as the pipeline makes them, in frames, with the session's fps beside
them, so a row turns back into the same Rep and the rule engine runs on it unchanged.
Seconds are for display only: frame / fps.

Each rep also keeps video_start_s, where it starts in the annotated video. That is a fact about
the rendered file, not a verdict. The file itself shows the verdicts of the day it was made:
if the thresholds change later, its pauses keep the old ones while the report shows the new.

A profile is a name, not an account: it keeps one person's sessions apart from another's,
it doesn't prove who anyone is.
"""

import math
import sqlite3
from dataclasses import astuple, dataclass, fields
from datetime import datetime
from pathlib import Path

from pipeline.models import Rep, SessionResult

# The reps table has one column per Rep field, generated from the dataclass, so the two can't
# drift apart. Names are quoted because one of them, "index", is an SQL keyword.
REP_FIELDS = fields(Rep)
SQL_TYPES = {int: "INTEGER", float: "REAL"}
REP_COLUMNS = ", ".join(f'"{f.name}"' for f in REP_FIELDS)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,          -- ISO 8601, UTC
    profile    TEXT NOT NULL,
    exercise   TEXT NOT NULL,
    video_id   TEXT NOT NULL UNIQUE,   -- the upload: its video and landmark cache
    fps        REAL NOT NULL,
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    n_frames   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_by_profile ON sessions (profile, created_at);
CREATE TABLE IF NOT EXISTS reps (
    session_id INTEGER NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    {", ".join(f'"{f.name}" {SQL_TYPES[f.type]}' for f in REP_FIELDS)},
    video_start_s REAL,              -- in the annotated video; NULL if it wasn't reached
    PRIMARY KEY (session_id, "index")
);
"""


@dataclass(frozen=True)
class Session:
    """One analysed video and its reps, as stored."""

    id: int
    created_at: str
    profile: str
    exercise: str
    video_id: str
    fps: float
    width: int
    height: int
    n_frames: int
    reps: tuple[Rep, ...]
    video_starts: dict[int, float | None]  # rep index -> seconds into the annotated video


def connect(path: Path) -> sqlite3.Connection:
    """Open the database, creating the tables on first use.

    One connection per request: sqlite3 connections can't be shared between threads, and
    FastAPI runs the endpoints in a thread pool. Opening a SQLite file is cheap.
    """
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row  # rows readable by column name
    db.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite, per connection
    db.executescript(SCHEMA)
    return db


def save_session(
    db: sqlite3.Connection,
    result: SessionResult,
    *,
    profile: str,
    exercise: str,
    video_id: str,
    created_at: datetime,
    video_starts: dict[int, float],
) -> int:
    """Store a session and its reps; return the session's id. video_starts: where each rep
    starts in the annotated video, by rep index.

    One transaction: `with db` commits if the block finishes and rolls back if anything in it
    raises, so a failure can't leave a session without its reps. Values go in through ?
    placeholders, never string formatting, so nothing in them can be run as SQL.
    """
    with db:
        cursor = db.execute(
            "INSERT INTO sessions"
            " (created_at, profile, exercise, video_id, fps, width, height, n_frames)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                created_at.isoformat(timespec="seconds"),
                profile,
                exercise,
                video_id,
                result.video.fps,
                result.video.width,
                result.video.height,
                result.video.n_frames,
            ),
        )
        session_id = cursor.lastrowid
        placeholders = ", ".join("?" * (len(REP_FIELDS) + 2))
        db.executemany(
            f"INSERT INTO reps (session_id, {REP_COLUMNS}, video_start_s) VALUES ({placeholders})",
            [(session_id, *_to_row(rep), video_starts.get(rep.index)) for rep in result.reps],
        )
    return session_id


def get_session(db: sqlite3.Connection, session_id: int) -> Session | None:
    """One session with its reps, or None if there is no such session."""
    row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return None if row is None else _session(row, *_reps(db, [session_id])[session_id])


def list_sessions(db: sqlite3.Connection, profile: str) -> list[Session]:
    """A profile's sessions with their reps, oldest first, as a progress chart reads them."""
    rows = db.execute(
        "SELECT * FROM sessions WHERE profile = ? ORDER BY created_at, id", (profile,)
    ).fetchall()
    reps = _reps(db, [row["id"] for row in rows])
    return [_session(row, *reps[row["id"]]) for row in rows]


def list_profiles(db: sqlite3.Connection) -> list[str]:
    """Every name that has a session, alphabetically."""
    return [row[0] for row in db.execute("SELECT DISTINCT profile FROM sessions ORDER BY 1")]


def _to_row(rep: Rep) -> tuple:
    # NaN ("could not measure") is stored as NULL, SQL's own "unknown"; SQLite would turn
    # it into NULL anyway, this says so.
    return tuple(None if isinstance(v, float) and math.isnan(v) else v for v in astuple(rep))


def _to_rep(row: sqlite3.Row) -> Rep:
    values = {}
    for f in REP_FIELDS:
        value = row[f.name]
        values[f.name] = math.nan if value is None and f.type is float else value
    return Rep(**values)


def _reps(
    db: sqlite3.Connection, session_ids: list[int]
) -> dict[int, tuple[tuple[Rep, ...], dict[int, float | None]]]:
    """The reps of several sessions in one query, by session id, each in rep order, with
    where each starts in the annotated video."""
    reps: dict[int, list[Rep]] = {session_id: [] for session_id in session_ids}
    starts: dict[int, dict[int, float | None]] = {session_id: {} for session_id in session_ids}
    if session_ids:
        marks = ", ".join("?" * len(session_ids))
        rows = db.execute(
            f'SELECT * FROM reps WHERE session_id IN ({marks}) ORDER BY session_id, "index"',
            session_ids,
        )
        for row in rows:
            reps[row["session_id"]].append(_to_rep(row))
            starts[row["session_id"]][row["index"]] = row["video_start_s"]
    return {session_id: (tuple(reps[session_id]), starts[session_id]) for session_id in reps}


def _session(
    row: sqlite3.Row, reps: tuple[Rep, ...], video_starts: dict[int, float | None]
) -> Session:
    columns = ("id", "created_at", "profile", "exercise", "video_id", "fps", "width", "height")
    return Session(
        *(row[column] for column in columns),
        n_frames=row["n_frames"],
        reps=reps,
        video_starts=video_starts,
    )
