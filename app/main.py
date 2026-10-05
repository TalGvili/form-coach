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
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.errors import AnalysisError, NotAVideoError
from pipeline.landmarks import probe_video
from pipeline.render import render
from pipeline.rules import load_rules
from pipeline.session import analyze, to_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pushup.yaml"
UPLOADS = ROOT / "data" / "uploads"
STATIC = Path(__file__).resolve().parent / "static"

# A server limit, not an analysis threshold. Analysing and rendering a 29 s clip took 51 s,
# so 60 s of video means a wait of nearly two minutes; a set of 10 push-ups lasts ~30 s.
MAX_DURATION_S = 60
VIDEO_ID = re.compile(r"[0-9a-f]{32}")  # the form of uuid4().hex
NO_REPS_HINT = (
    "No push-ups found. Film from the side with your whole body in the frame, and start "
    "recording before you get down."
)

app = FastAPI(title="Form Coach")
# Read once at startup, so a broken config stops the server instead of failing every upload.
TITLES = {rule.name: rule.title for rule in load_rules(CONFIG)}


@app.exception_handler(AnalysisError)
def refuse(request: Request, error: AnalysisError) -> JSONResponse:
    """Every video the pipeline refuses, turned into a response in one place. The message is
    already written for the user. Not a video at all is a bad request (400); a video that
    can't be analysed is 422, "understood but can't be processed"."""
    status = 400 if isinstance(error, NotAVideoError) else 422
    # {"detail": ...}, the same shape as HTTPException's, so the page reads one field
    return JSONResponse(status_code=status, content={"detail": str(error)})


@app.post("/analyze")
def analyze_upload(video: UploadFile) -> dict[str, Any]:
    """Save the upload, check it, analyse it and render the annotated video; return the report."""
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
    render(result, TITLES, UPLOADS / f"{video_id}_annotated.mp4")

    report = to_json(result)
    del report["video"]["path"]  # a path on this server, of no use to the user
    report["id"] = video_id
    report["video_url"] = f"/videos/{video_id}"
    report["titles"] = TITLES
    report["hint"] = None if result.reps else NO_REPS_HINT
    return report


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
