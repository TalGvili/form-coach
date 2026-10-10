"""FastAPI app: upload endpoint, annotated video serving, static frontend.

No analysis logic here - this layer only calls into `pipeline` and shapes HTTP responses.

Two modes. With GOOGLE_CLIENT_ID set, people sign in with Google (app/auth.py) and every request
works on the signed-in user's own data. Without it, the app runs as a local, single-computer app
whose profiles are names, not accounts. owner_of() is the one place that decides whose data a
request is about.

    uvicorn app.main:app --reload --host 0.0.0.0

The endpoints are plain `def`, not `async def`. FastAPI runs a plain def in a pool of worker
threads, so a slow analysis (MediaPipe takes about as long as the video) occupies one thread
while the server keeps answering other requests. An async def runs on the event loop itself,
the one thread that serves every request, and slow code there would freeze them all.
"""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import yaml
from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import ASGIApp, Receive, Scope, Send

from app import db as history
from app.auth import SignInError, new_login_token, token_hash, verify_google_token
from pipeline.errors import AnalysisError, NotAVideoError
from pipeline.landmarks import cache_path, probe_video
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

# Sign in with Google, on when the client id is set (it isn't a secret: it's in every page).
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID") or None


def allowed_emails(setting: str) -> set[str] | None:
    """Who may sign in, from ALLOWED_EMAILS: a comma-separated list, or "*" for any Google
    account (None). Deny by default: an empty setting allows nobody, and the server refuses to
    start that way, so letting every Google account in is always a written-out choice.

    Google's own test-user list for an app in Testing mode doesn't stop this kind of sign-in
    (only name and email, no access to Google services): a test showed another account getting
    in. This list, checked by this server, is the control."""
    if setting.strip() == "*":
        return None
    return {email.strip().lower() for email in setting.split(",") if email.strip()}


ALLOWED_EMAILS = allowed_emails(os.environ.get("ALLOWED_EMAILS", ""))
LOGIN_COOKIE = "form_coach_login"
LOGIN_DAYS = 30

# A server limit, not an analysis threshold. Analysing and rendering a 29 s clip took 51 s,
# so 60 s of video means a wait of nearly two minutes; a set of 10 push-ups lasts ~30 s.
MAX_DURATION_S = 60
# Bytes, against filling the disk. 30 s of 1080p phone video is ~34 MB; a minute of 4K is up to
# ~450 MB, so 500 MB refuses nothing the duration limit would accept.
MAX_UPLOAD_MB = 500
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
    if GOOGLE_CLIENT_ID and ALLOWED_EMAILS == set():
        raise RuntimeError(
            "Google sign-in is on but nobody may sign in: set ALLOWED_EMAILS to the emails that "
            "may (comma-separated), or ALLOWED_EMAILS=* to allow any Google account."
        )
    yield


class LimitUploadSize:
    """Refuse a request body over max_bytes before any of it is read.

    FastAPI reads a whole upload to a temporary file before the endpoint runs, so a size check
    in the endpoint comes after the bytes are already on disk. This middleware sees each request
    first and reads its declared size, the Content-Length header. uvicorn never passes on more
    body than that header declared, so it can't understate. A body without one is refused too:
    browsers always send it with a file.

    Middleware runs on the event loop, so it is `async def`; it only reads a header, nothing slow.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            declared = dict(scope["headers"]).get(b"content-length")
            if declared is None or int(declared) > self.max_bytes:
                size = f"{self.max_bytes // 1024**2} MB"
                refusal = JSONResponse(
                    status_code=413 if declared else 411,
                    content={"detail": f"Uploads must be under {size}."},
                )
                await refusal(scope, receive, send)
                return
        await self.app(scope, receive, send)


app = FastAPI(title="Form Coach", lifespan=lifespan)
app.add_middleware(LimitUploadSize, max_bytes=MAX_UPLOAD_MB * 1024**2)
# Read once at startup, so a broken config stops the server instead of failing every upload.
RULES = load_rules(CONFIG)
TITLES = {rule.name: rule.title for rule in RULES}
EXERCISE = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["exercise"]
# Progress per metric covers the rules the page charts, so no metric is named in code.
CHARTED = {rule.metric: rule.title for rule in RULES if rule.chart}

# Local mode's profile, a name, not an account (see app/db.py): at least one visible character,
# so "   " can't become one. Optional, because with Google sign-in the request doesn't choose.
Profile = Annotated[str | None, Form(min_length=1, max_length=40, pattern=r"\S")]


def open_history() -> closing[sqlite3.Connection]:
    """A connection for one request, closed when the `with` block ends. closing() is needed
    because a sqlite3 connection's own `with` commits a transaction but doesn't close it."""
    return closing(history.connect(DB_PATH))


def signed_in_user(request: Request) -> history.User | None:
    """The user whose login cookie this request carries, if it's valid and not expired."""
    token = request.cookies.get(LOGIN_COOKIE)
    if not token:
        return None
    with open_history() as db:
        return history.user_for_login(db, token_hash(token), datetime.now(UTC))


def owner_of(request: Request, profile: str | None = None) -> str:
    """Whose data a request reads or writes: the one place that decides it.

    With Google sign-in, always the signed-in user's, whatever the request names, so nobody can
    ask for someone else's sessions by changing a parameter. Without it, the profile the request
    names: local mode has no accounts to check against.
    """
    if GOOGLE_CLIENT_ID:
        user = signed_in_user(request)
        if user is None:
            raise HTTPException(401, "Sign in first.")
        return user.owner
    if profile is None or not profile.strip():
        raise HTTPException(422, "Choose a profile first.")
    return profile.strip()


def profiles_mode_only() -> None:
    """Profiles are local mode's stand-in for accounts; with sign-in on they don't exist."""
    if GOOGLE_CLIENT_ID:
        raise HTTPException(404, "This server uses Google sign-in, not profiles.")


@app.exception_handler(AnalysisError)
def refuse(request: Request, error: AnalysisError) -> JSONResponse:
    """Every video the pipeline refuses, turned into a response in one place. The message is
    already written for the user. Not a video at all is a bad request (400); a video that
    can't be analysed is 422, "understood but can't be processed"."""
    status = 400 if isinstance(error, NotAVideoError) else 422
    # {"detail": ...}, the same shape as HTTPException's, so the page reads one field
    return JSONResponse(status_code=status, content={"detail": str(error)})


@app.post("/analyze")
def analyze_upload(request: Request, video: UploadFile, profile: Profile = None) -> dict[str, Any]:
    """Save the upload, check it, analyse it, render the annotated video and save the session
    to its owner's history; return the report."""
    owner = owner_of(request, profile)  # before any work: no sign-in, no analysis
    video_id = uuid.uuid4().hex
    # The file name is ours, never the uploader's: a random id can't collide with another
    # upload's landmark cache (cached by file name) or point outside UPLOADS. OpenCV reads the
    # format from the file's contents, so a .mov saved as .mp4 still opens.
    path = UPLOADS / f"{video_id}.mp4"
    annotated = UPLOADS / f"{video_id}_annotated.mp4"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as saved:
        shutil.copyfileobj(video.file, saved)

    try:
        info = probe_video(path)  # milliseconds, where MediaPipe takes as long as the video
        if info.n_frames / info.fps > MAX_DURATION_S:
            raise HTTPException(413, f"Videos must be under {MAX_DURATION_S} seconds.")
        result = analyze(path, CONFIG)
        with h264_writer(annotated, result.video) as write:
            starts = annotate(result, TITLES, write)
    except Exception:
        # refused or failed: nothing of this upload is kept
        annotated.unlink(missing_ok=True)
        cache_path(video_id).unlink(missing_ok=True)
        raise
    finally:
        # The original is read only while drawing the annotated video. History keeps the
        # measurements, the annotated video and the landmark cache, so it's never needed again,
        # and a video of someone's body shouldn't sit on disk for no reason.
        path.unlink(missing_ok=True)

    session_id = None
    with open_history() as db:
        now = datetime.now(UTC)
        if not GOOGLE_CLIENT_ID:
            # the profile's stored spelling: "TAL" still goes into Tal's history
            owner = history.add_profile(db, owner, now)
        history.add_video(db, video_id, owner, now)
        if result.reps:  # no reps is a filming problem, not a session worth tracking
            session_id = history.save_session(
                db,
                result,
                profile=owner,
                exercise=EXERCISE,
                video_id=video_id,
                created_at=now,
                video_starts=starts,
            )
    return report_for(result, video_id, starts, session_id)


class GoogleCredential(BaseModel):
    """The body of POST /auth/google: the ID token Google's button gave the page."""

    credential: str


@app.post("/auth/google")
def sign_in_with_google(
    body: GoogleCredential, request: Request, response: Response
) -> dict[str, str]:
    """Check Google's ID token and start a login: a random token in a cookie.

    The body must be JSON (FastAPI rejects a form with 422). Another website can't send JSON to
    this server without the browser first asking this server's permission, which it never gives,
    so no other page can sign a visitor in to an account of its choosing.
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(404, "Google sign-in isn't set up on this server.")
    try:
        identity = verify_google_token(body.credential, GOOGLE_CLIENT_ID)
    except SignInError as error:
        raise HTTPException(401, "Google sign-in didn't work. Try again.") from error
    if ALLOWED_EMAILS is not None and (
        not identity.email_verified or identity.email.lower() not in ALLOWED_EMAILS
    ):
        raise HTTPException(403, "This Google account isn't allowed on this server.")

    now = datetime.now(UTC)
    token = new_login_token()
    with open_history() as db:
        user = history.upsert_user(db, identity.sub, identity.email, identity.name, now)
        history.start_login(db, user, token_hash(token), now, now + timedelta(days=LOGIN_DAYS))
    # HttpOnly: the page's scripts can't read it, so injected script can't steal it. SameSite=Lax:
    # the browser doesn't send it with another site's POST, so other sites can't act as the user.
    # Secure over HTTPS: never sent unencrypted. (Plain-HTTP localhost has no HTTPS to require.)
    response.set_cookie(
        LOGIN_COOKIE,
        token,
        max_age=LOGIN_DAYS * 24 * 3600,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return {"name": user.name, "email": user.email}


@app.get("/me")
def me(request: Request) -> dict[str, str]:
    """Who is signed in; 401 if nobody. The page asks on load, since it can't read the cookie."""
    user = signed_in_user(request) if GOOGLE_CLIENT_ID else None
    if user is None:
        raise HTTPException(401, "Not signed in.")
    return {"name": user.name, "email": user.email}


@app.post("/auth/logout")
def sign_out(request: Request, response: Response) -> dict[str, bool]:
    """End the login on the server, so the token stops working even if a copy survives."""
    token = request.cookies.get(LOGIN_COOKIE)
    if token:
        with open_history() as db:
            history.end_login(db, token_hash(token))
    response.delete_cookie(LOGIN_COOKIE)
    return {"signed_out": True}


@app.get("/config.js")
def page_config() -> Response:
    """Tells the pages, before their own scripts run, whether Google sign-in is on and with
    which client id. A script rather than JSON, so a page knows its mode without waiting."""
    config = {"googleClientId": GOOGLE_CLIENT_ID}
    return Response(
        f"window.FORM_COACH = {json.dumps(config)};\n",
        media_type="text/javascript",
        headers={"Cache-Control": "no-store"},  # a restart with sign-in turned on shows at once
    )


@app.get("/profiles")
def profiles() -> list[str]:
    """Every profile, for the landing page's tiles."""
    profiles_mode_only()
    with open_history() as db:
        return history.list_profiles(db)


@app.post("/profiles")
def add_profile(request: Request, profile: Profile = None) -> dict[str, str]:
    """Create a profile, or find the existing one with that name in any capitals; returns the
    name to use from now on."""
    profiles_mode_only()
    name = owner_of(request, profile)
    with open_history() as db:
        return {"profile": history.add_profile(db, name, datetime.now(UTC))}


@app.get("/sessions")
def sessions(request: Request, profile: str | None = None) -> list[dict[str, Any]]:
    """The owner's sessions, oldest first, each judged by the current rules."""
    owner = owner_of(request, profile)
    with open_history() as db:
        stored = history.list_sessions(db, owner)
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
def progress(request: Request, profile: str | None = None) -> dict[str, Any]:
    """The numbers behind the history page's chart: one entry per session of the owner,
    oldest first, each computed now from the stored measurements and the current rules.
    metrics: the per-metric fields' keys, with the titles to show them under."""
    owner = owner_of(request, profile)
    with open_history() as db:
        stored = history.list_sessions(db, owner)
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
def session_report(request: Request, session_id: int) -> dict[str, Any]:
    """A stored session's report, in the same shape as POST /analyze returns, with its
    faults recomputed from the stored measurements and the current rules.

    With sign-in on, someone else's session is "no such session", not "forbidden": a different
    answer would tell a stranger which ids exist."""
    with open_history() as db:
        session = history.get_session(db, session_id)
    if session is None or (GOOGLE_CLIENT_ID and session.profile != owner_of(request)):
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
def annotated_video(request: Request, video_id: str) -> FileResponse:
    """The annotated video of one upload. The id must look like one this server issued, so a
    request can't name a file of its own choosing, such as ../../configs/pushup.yaml. With
    sign-in on, it must also be the signed-in user's: 404 otherwise, as for sessions."""
    path = UPLOADS / f"{video_id}_annotated.mp4"
    if not VIDEO_ID.fullmatch(video_id) or not path.is_file():
        raise HTTPException(404, "No such video.")
    if GOOGLE_CLIENT_ID:
        with open_history() as db:
            if history.video_owner(db, video_id) != owner_of(request):
                raise HTTPException(404, "No such video.")
    return FileResponse(path, media_type="video/mp4")


# Mounted last: a mount at "/" matches every path, so routes added after it would be hidden.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
