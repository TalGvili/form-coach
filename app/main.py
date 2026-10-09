"""FastAPI app: upload endpoint, annotated video serving, static frontend.

No analysis logic here - this layer only calls into `pipeline` and shapes HTTP responses.

    uvicorn app.main:app --reload --host 0.0.0.0

The endpoints are plain `def`, not `async def`. FastAPI runs a plain def in a pool of worker
threads, so a slow analysis (MediaPipe takes about as long as the video) occupies one thread
while the server keeps answering other requests. An async def runs on the event loop itself,
the one thread that serves every request, and slow code there would freeze them all.
"""

import re
import shutil
import sqlite3
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import yaml
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import db as history
from pipeline.errors import AnalysisError, NotAVideoError
from pipeline.landmarks import probe_video
from pipeline.models import SessionResult, VideoInfo
from pipeline.progress import session_progress
from pipeline.render import annotate
from pipeline.rules import load_rules
from pipeline.session import analyze, judge, to_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pushup.yaml"
UPLOADS = ROOT / "data" / "uploads"
DB_PATH = ROOT / "data" / "history.db"
STATIC = Path(__file__).resolve().parent / "static"

# A server limit, not an analysis threshold. Analysing and rendering a 29 s clip took 51 s,
# so 60 s of video means a wait of nearly two minutes; a set of 10 push-ups lasts ~30 s.
MAX_DURATION_S = 60
VIDEO_ID = re.compile(r"[0-9a-f]{32}")  # the form of uuid4().hex
NO_REPS_HINT = (
    "No push-ups found. Film from the side with your whole body in the frame, and start "
    "recording before you get down."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once as the server starts. Without ffmpeg every upload would fail, but only after a
    minute of analysis, so refuse to start instead."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; it re-encodes videos for the browser.")
    yield


app = FastAPI(title="Form Coach", lifespan=lifespan)
# Read once at startup, so a broken config stops the server instead of failing every upload.
RULES = load_rules(CONFIG)
TITLES = {rule.name: rule.title for rule in RULES}
EXERCISE = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["exercise"]
# Progress per metric covers the rules the page charts, so no metric is named in code.
CHARTED = {rule.metric: rule.title for rule in RULES if rule.chart}

# Who a session belongs to: a name, not an account (see app/db.py). At least one visible
# character, so "   " can't become a profile.
Profile = Annotated[str, Form(min_length=1, max_length=40, pattern=r"\S")]


def open_history() -> closing[sqlite3.Connection]:
    """A connection for one request, closed when the `with` block ends. closing() is needed
    because a sqlite3 connection's own `with` commits a transaction but doesn't close it."""
    return closing(history.connect(DB_PATH))


@app.exception_handler(AnalysisError)
def refuse(request: Request, error: AnalysisError) -> JSONResponse:
    """Every video the pipeline refuses, turned into a response in one place. The message is
    already written for the user. Not a video at all is a bad request (400); a video that
    can't be analysed is 422, "understood but can't be processed"."""
    status = 400 if isinstance(error, NotAVideoError) else 422
    # {"detail": ...}, the same shape as HTTPException's, so the page reads one field
    return JSONResponse(status_code=status, content={"detail": str(error)})


@app.post("/analyze")
def analyze_upload(video: UploadFile, profile: Profile) -> dict[str, Any]:
    """Save the upload, check it, analyse it, render the annotated video and save the session
    to the profile's history; return the report."""
    video_id = uuid.uuid4().hex
    # The file name is ours, never the uploader's: a random id can't collide with another
    # upload's landmark cache (cached by file name) or point outside UPLOADS. OpenCV reads the
    # format from the file's contents, so a .mov saved as .mp4 still opens.
    path = UPLOADS / f"{video_id}.mp4"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as saved:
        shutil.copyfileobj(video.file, saved)

    info = probe_video(path)  # milliseconds, where MediaPipe takes as long as the video
    if info.n_frames / info.fps > MAX_DURATION_S:
        raise HTTPException(413, f"Videos must be under {MAX_DURATION_S} seconds.")

    result = analyze(path, CONFIG)
    with h264_writer(UPLOADS / f"{video_id}_annotated.mp4", result.video) as write:
        starts = annotate(result, TITLES, write)

    session_id = None
    if result.reps:  # no reps is a filming problem, not a session worth tracking
        with open_history() as db:
            now = datetime.now(UTC)
            session_id = history.save_session(
                db,
                result,
                # the profile's stored spelling: "TAL" still goes into Tal's history
                profile=history.add_profile(db, profile.strip(), now),
                exercise=EXERCISE,
                video_id=video_id,
                created_at=now,
                video_starts=starts,
            )
    return report_for(result, video_id, starts, session_id)


@app.get("/profiles")
def profiles() -> list[str]:
    """Every profile, for the landing page's tiles."""
    with open_history() as db:
        return history.list_profiles(db)


@app.post("/profiles")
def add_profile(profile: Profile) -> dict[str, str]:
    """Create a profile, or find the existing one with that name in any capitals; returns the
    name to use from now on."""
    with open_history() as db:
        return {"profile": history.add_profile(db, profile.strip(), datetime.now(UTC))}


@app.get("/sessions")
def sessions(profile: str) -> list[dict[str, Any]]:
    """A profile's sessions, oldest first, each judged by the current rules."""
    with open_history() as db:
        stored = history.list_sessions(db, profile)
    summaries = []
    for session in stored:
        progress = session_progress(judged(session), [])
        summaries.append(
            {
                "id": session.id,
                "created_at": session.created_at,
                "reps": progress.reps,
                "to_fix": progress.to_fix,
            }
        )
    return summaries


@app.get("/progress")
def progress(profile: str) -> dict[str, Any]:
    """The numbers behind the history page's chart: one entry per session of the profile,
    oldest first, each computed now from the stored measurements and the current rules.
    metrics: the per-metric fields' keys, with the titles to show them under."""
    with open_history() as db:
        stored = history.list_sessions(db, profile)
    return {
        "metrics": CHARTED,
        "sessions": [
            {
                "id": session.id,
                "created_at": session.created_at,
                **to_json(session_progress(judged(session), list(CHARTED))),
            }
            for session in stored
        ],
    }


@app.get("/sessions/{session_id}")
def session_report(session_id: int) -> dict[str, Any]:
    """A stored session's report, in the same shape as POST /analyze returns, with its
    faults recomputed from the stored measurements and the current rules."""
    with open_history() as db:
        session = history.get_session(db, session_id)
    if session is None:
        raise HTTPException(404, "No such session.")
    report = report_for(judged(session), session.video_id, session.video_starts, session.id)
    report["created_at"] = session.created_at
    return report


def judged(session: history.Session) -> SessionResult:
    """A stored session's reps, judged now: verdicts are never stored."""
    video = VideoInfo(
        path=session.video_id,
        fps=session.fps,
        width=session.width,
        height=session.height,
        n_frames=session.n_frames,
    )
    return judge(video, list(session.reps), RULES)


def report_for(
    result: SessionResult,
    video_id: str,
    starts: dict[int, float | None],
    session_id: int | None,
) -> dict[str, Any]:
    """The JSON the page reads, for a new analysis and for a stored session alike."""
    report = to_json(result)
    del report["video"]["path"]  # a path on this server, of no use to the user
    for rep in report["reps"]:
        # where the rep starts in the annotated video, which runs longer than the original
        rep["video_start_s"] = starts.get(rep["index"])
    report["id"] = video_id
    report["session_id"] = session_id  # None when it wasn't saved (no reps)
    report["video_url"] = f"/videos/{video_id}"
    # what the page needs to label each check and draw its limit; the numbers stay in the YAML
    report["rules"] = {
        r.name: {"title": r.title, "metric": r.metric, "min": r.min, "max": r.max, "chart": r.chart}
        for r in RULES
    }
    report["hint"] = None if result.reps else NO_REPS_HINT
    return report


@contextmanager
def h264_writer(out: Path, video: VideoInfo) -> Iterator[Callable[[np.ndarray], None]]:
    """A write(frame) function that pipes frames into ffmpeg, which encodes them as H.264,
    the codec every browser plays.

    Each frame is encoded once. Writing an mp4v file with OpenCV and re-encoding it took
    35 s for a 21 s clip, 28 of them in OpenCV's encoder; piping takes 15 s for the same
    file. The input options describe the raw frames (OpenCV's BGR pixels, size, frame rate);
    "-i -" reads them from stdin.

    Output: yuv420p is the pixel format browsers and phones expect; faststart moves the index
    to the front so playback starts before the download finishes; the scale caps the height
    at 720 px, plenty on a phone (-2 keeps the proportions with an even width, which H.264
    needs); -an, no sound. veryfast with crf 26 encoded a 33 s annotated clip in 8 s at 17 MB,
    against 22 s and 33 MB at the defaults, with the opening card's text looking the same.
    """
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24"]
    command += ["-s", f"{video.width}x{video.height}", "-r", str(video.fps), "-i", "-"]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"]
    command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    command += ["-vf", "scale=-2:'min(720,ih)'", "-an", str(out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        yield lambda frame: process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()  # end of input: ffmpeg finishes the file and exits
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


@app.get("/videos/{video_id}")
def annotated_video(video_id: str) -> FileResponse:
    """The annotated video of one upload. The id must look like one this server issued, so a
    request can't name a file of its own choosing, such as ../../configs/pushup.yaml."""
    path = UPLOADS / f"{video_id}_annotated.mp4"
    if not VIDEO_ID.fullmatch(video_id) or not path.is_file():
        raise HTTPException(404, "No such video.")
    return FileResponse(path, media_type="video/mp4")


# Mounted last: a mount at "/" matches every path, so routes added after it would be hidden.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
