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

Every session and upload has an owner, a string: in local mode a profile name (a name, not an
account: it keeps people's sessions apart, it doesn't prove who anyone is); with Google sign-in,
"google:<sub>" for the signed-in user (User.owner). The history code doesn't care which.

Login tokens are stored as SHA-256 hashes (app/auth.py says why), with an expiry.
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
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY,
    google_sub TEXT NOT NULL UNIQUE,   -- Google's permanent id for the account
    email      TEXT NOT NULL,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS logins (
    token_hash TEXT PRIMARY KEY,       -- SHA-256 of the cookie's token, never the token itself
    user_id    INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS videos (
    video_id   TEXT PRIMARY KEY,       -- every annotated video, saved session or not
    owner      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (
    name       TEXT PRIMARY KEY COLLATE NOCASE,  -- "tal" and "Tal" are the same person
    created_at TEXT NOT NULL
);
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


def add_profile(db: sqlite3.Connection, name: str, created_at: datetime) -> str:
    """Create a profile unless one with that name exists, ignoring case; return the name as
    stored, so "tal" typed later still means the existing "Tal"."""
    with db:
        db.execute(
            "INSERT OR IGNORE INTO profiles (name, created_at) VALUES (?, ?)",
            (name, created_at.isoformat(timespec="seconds")),
        )
    return db.execute("SELECT name FROM profiles WHERE name = ?", (name,)).fetchone()[0]


def list_profiles(db: sqlite3.Connection) -> list[str]:
    """Every profile, alphabetically: those added, and any name with sessions saved before
    profiles had their own table."""
    rows = db.execute(
        "SELECT name FROM profiles UNION SELECT profile FROM sessions ORDER BY 1 COLLATE NOCASE"
    )
    return [row[0] for row in rows]


@dataclass(frozen=True)
class User:
    """Someone signed in with Google."""

    id: int
    google_sub: str
    email: str
    name: str

    @property
    def owner(self) -> str:
        """What their sessions and videos are stored under."""
        return f"google:{self.google_sub}"


def upsert_user(db: sqlite3.Connection, sub: str, email: str, name: str, now: datetime) -> User:
    """The user with this Google id, created on their first sign-in. Email and name are
    refreshed every time, since they can change on Google's side; the id can't."""
    with db:
        db.execute(
            "INSERT INTO users (google_sub, email, name, created_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT (google_sub) DO UPDATE SET email = excluded.email, name = excluded.name",
            (sub, email, name, now.isoformat(timespec="seconds")),
        )
    row = db.execute("SELECT * FROM users WHERE google_sub = ?", (sub,)).fetchone()
    return User(row["id"], row["google_sub"], row["email"], row["name"])


def start_login(
    db: sqlite3.Connection, user: User, token_hash: str, now: datetime, expires: datetime
) -> None:
    with db:
        db.execute(
            "INSERT INTO logins (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token_hash, user.id, now.isoformat(), expires.isoformat()),
        )


def user_for_login(db: sqlite3.Connection, token_hash: str, now: datetime) -> User | None:
    """The user a login token belongs to, or None if it's unknown, ended or expired."""
    row = db.execute(
        "SELECT users.* FROM logins JOIN users ON users.id = logins.user_id"
        " WHERE logins.token_hash = ? AND logins.expires_at > ?",
        (token_hash, now.isoformat()),
    ).fetchone()
    return None if row is None else User(row["id"], row["google_sub"], row["email"], row["name"])


def end_login(db: sqlite3.Connection, token_hash: str) -> None:
    """Sign out: the token stops working at once, whatever the cookie's own expiry says."""
    with db:
        db.execute("DELETE FROM logins WHERE token_hash = ?", (token_hash,))


def add_video(db: sqlite3.Connection, video_id: str, owner: str, now: datetime) -> None:
    """Record who an annotated video belongs to, so only they can fetch it."""
    with db:
        db.execute(
            "INSERT INTO videos (video_id, owner, created_at) VALUES (?, ?, ?)",
            (video_id, owner, now.isoformat(timespec="seconds")),
        )


def video_owner(db: sqlite3.Connection, video_id: str) -> str | None:
    row = db.execute("SELECT owner FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    return None if row is None else row["owner"]


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
