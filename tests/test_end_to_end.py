"""One real upload, end to end: MediaPipe, the rules, ffmpeg and the database, nothing faked.

Every other test fakes the pose model, so none of them would notice a broken model download, a
MediaPipe update that changes its output, or a system library missing in the Docker image. This
one needs the model and ffmpeg and is skipped without them: it runs locally and inside the
Docker image in CI, not in the plain test job.

The fixture is the first 12.6 s of clip09, scaled to 360p: the clip the public demo shows, three
reps, the second one shallow. Only outcomes far from every threshold are checked (depth 4-7 deg
on the deep reps and 32 on the shallow one, limit 15; lockout 166 deg and up, limit 156), because
the same video gives landmarks a few pixels apart on Windows and Linux.
"""

import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from pipeline import landmarks

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "three_pushups.mp4"

# A skipped test passes, so in the Docker image, where the model and ffmpeg must be present,
# CI sets this and a missing one fails the test instead of skipping it.
REQUIRED = os.environ.get("FORM_COACH_E2E") == "required"

pytestmark = pytest.mark.skipif(
    not REQUIRED and (not landmarks.MODEL_PATH.exists() or shutil.which("ffmpeg") is None),
    reason="needs the pose model and ffmpeg",
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The real app in local mode, writing its uploads, cache and history to a temporary folder."""
    monkeypatch.setattr(main, "GOOGLE_CLIENT_ID", None)
    monkeypatch.setattr(main, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(landmarks, "CACHE_DIR", tmp_path / "cache")
    return TestClient(main.app)


def test_a_real_video_is_analysed_annotated_and_saved(client: TestClient):
    with FIXTURE.open("rb") as video:
        response = client.post(
            "/analyze",
            files={"video": ("three_pushups.mp4", video, "video/mp4")},
            data={"profile": "Test"},
        )
    assert response.status_code == 200, response.text
    report = response.json()

    assert [rep["index"] for rep in report["reps"]] == [1, 2, 3]
    shown = [(f["rep_index"], f["rule"]) for f in report["faults"] if not f["suppressed"]]
    assert shown == [(2, "shallow")]
    assert report["unevaluated"] == []

    # the annotated video plays from each rep's start
    assert all(rep["video_start_s"] is not None for rep in report["reps"])
    video = client.get(report["video_url"])
    assert video.status_code == 200
    assert video.content[4:8] == b"ftyp"  # an MP4 file starts with this box

    # and the session is in the history
    sessions = client.get("/sessions", params={"profile": "Test"}).json()
    assert [s["id"] for s in sessions] == [report["session_id"]]
