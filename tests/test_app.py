"""Tests for app.main, the HTTP layer only.

analyze, render and probe_video are replaced by fakes, so no model, video or MediaPipe is
needed. What is tested is what this layer owns: status codes, messages, and the response
shape. The text-file test uses the real probe_video, since refusing a non-video is its job.
"""

import shutil
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from pipeline.errors import SingleRepError
from pipeline.landmarks import probe_video
from pipeline.models import Rep, SessionResult, VideoInfo

INFO = VideoInfo(path="uploaded.mp4", fps=30.0, width=640, height=360, n_frames=300)
NAN = float("nan")
REP = Rep(1, 0, 30, 60, NAN, *[0.0] * 6, *[0] * 5)  # min_elbow_angle couldn't be measured


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose uploads go to a temporary folder, with a 10 s video that analyses to one
    rep. A test changes the fakes it cares about."""
    monkeypatch.setattr(main, "UPLOADS", tmp_path)
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(main, "probe_video", lambda path: INFO)
    monkeypatch.setattr(main, "analyze", lambda path, config: SessionResult(INFO, [REP], [], []))

    def fake_annotate(result, titles, write):
        write(b"video")
        return {1: 3.5}

    monkeypatch.setattr(main, "annotate", fake_annotate)

    @contextmanager
    def file_writer(out, video):
        yield out.write_bytes

    monkeypatch.setattr(main, "h264_writer", file_writer)
    return TestClient(main.app)


def upload(client: TestClient, content: bytes = b"any bytes", profile: str = "tal"):
    return client.post(
        "/analyze",
        files={"video": ("set.mp4", content, "video/mp4")},
        data={"profile": profile},
    )


def test_a_video_returns_the_report_and_a_link_to_the_annotated_video(client: TestClient):
    response = upload(client)
    assert response.status_code == 200
    report = response.json()
    assert len(report["reps"]) == 1
    assert report["reps"][0]["min_elbow_angle"] is None  # NaN became null: valid JSON
    assert "path" not in report["video"]  # the server's file layout stays private
    assert report["hint"] is None
    assert report["reps"][0]["video_start_s"] == 3.5
    assert report["rules"]["shallow"]["title"] == "Depth"
    video = client.get(report["video_url"])
    assert video.status_code == 200
    assert video.content == b"video"


def test_no_reps_comes_with_a_hint(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "analyze", lambda path, config: SessionResult(INFO, [], [], []))
    response = upload(client)
    assert response.status_code == 200
    assert response.json()["hint"] == main.NO_REPS_HINT


def test_a_file_that_is_not_a_video_is_a_bad_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(main, "probe_video", probe_video)  # the real one
    response = upload(client, b"just some text")
    assert response.status_code == 400
    assert response.json()["detail"] == "Please upload a video file."


def test_a_video_over_the_limit_is_refused_before_analysis(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    long_video = VideoInfo("long.mp4", 30.0, 640, 360, 30 * (main.MAX_DURATION_S + 1))
    monkeypatch.setattr(main, "probe_video", lambda path: long_video)

    def must_not_run(path, config):
        raise AssertionError("analysed a video that should have been refused")

    monkeypatch.setattr(main, "analyze", must_not_run)
    assert upload(client).status_code == 413


def test_a_refused_video_returns_the_pipelines_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    def single_rep(path, config):
        raise SingleRepError("Only one rep found. Film at least two reps to be analysed.")

    monkeypatch.setattr(main, "analyze", single_rep)
    response = upload(client)
    assert response.status_code == 422
    assert response.json()["detail"].startswith("Only one rep found")


@pytest.mark.parametrize("video_id", ["0" * 32, "..%2F..%2Fconfigs%2Fpushup.yaml", "abc"])
def test_an_unknown_or_malformed_video_id_is_not_found(client: TestClient, video_id: str):
    assert client.get(f"/videos/{video_id}").status_code == 404


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_frames_are_piped_into_an_h264_video(tmp_path: Path):
    """The real ffmpeg, fed ten small frames."""
    out = tmp_path / "out.mp4"
    video = VideoInfo("small.mp4", fps=30.0, width=320, height=240, n_frames=10)
    with main.h264_writer(out, video) as write:
        for shade in range(0, 250, 25):
            write(np.full((240, 320, 3), shade, dtype=np.uint8))

    capture = cv2.VideoCapture(str(out))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    assert fourcc in ("avc1", "h264")
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240  # smaller than 720: not enlarged
    capture.release()


@pytest.mark.parametrize("profile", ["", "   ", "x" * 41])
def test_an_upload_needs_a_name_to_save_it_under(client: TestClient, profile: str):
    assert upload(client, profile=profile).status_code == 422


def test_an_analysis_is_saved_to_its_profiles_history(client: TestClient):
    session_id = upload(client, profile=" tal ").json()["session_id"]
    assert client.get("/profiles").json() == ["tal"]  # stored without the spaces
    listed = client.get("/sessions", params={"profile": "tal"}).json()
    assert [s["id"] for s in listed] == [session_id]
    assert client.get("/sessions", params={"profile": "dana"}).json() == []


def test_a_stored_session_is_judged_by_the_current_rules(client: TestClient):
    """The fake analysis reports no faults, but REP's lockout angle (0 deg) is below the
    config's minimum. Read back from history, the rep is judged again and the fault appears:
    measurements are stored, verdicts are computed when read."""
    report = upload(client).json()
    assert report["faults"] == []
    stored = client.get(f"/sessions/{report['session_id']}").json()
    assert [f["rule"] for f in stored["faults"]] == ["no_lockout"]
    assert stored["reps"][0]["video_start_s"] == 3.5
    assert stored["video_url"] == report["video_url"]
    assert stored["created_at"]  # the page shows when it was filmed


def test_a_video_without_reps_is_not_saved(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "analyze", lambda path, config: SessionResult(INFO, [], [], []))
    assert upload(client).json()["session_id"] is None
    assert client.get("/profiles").json() == []


def test_an_unknown_session_is_not_found(client: TestClient):
    assert client.get("/sessions/99").status_code == 404


def test_progress_has_one_entry_per_session_with_the_charted_metrics(client: TestClient):
    upload(client)
    upload(client)
    progress = client.get("/progress", params={"profile": "tal"}).json()
    assert progress["metrics"] == {"min_upper_arm_angle": "Depth"}
    assert len(progress["sessions"]) == 2
    first = progress["sessions"][0]
    # REP (0 deg lockout) is a no_lockout fault under the current rules: judged when read
    assert (first["reps"], first["to_fix"], first["clean_share"]) == (1, 1, 0.0)
    assert first["average"] == {"min_upper_arm_angle": 0.0}
    assert first["fatigue"] == {"min_upper_arm_angle": None}  # one rep has no thirds: null


def test_adding_a_profile_returns_the_name_to_use(client: TestClient):
    assert client.post("/profiles", data={"profile": " Tal "}).json() == {"profile": "Tal"}
    assert client.post("/profiles", data={"profile": "TAL"}).json() == {"profile": "Tal"}
    assert client.get("/profiles").json() == ["Tal"]
    assert client.post("/profiles", data={"profile": "  "}).status_code == 422


def test_an_upload_uses_the_profiles_stored_spelling(client: TestClient):
    client.post("/profiles", data={"profile": "Tal"})
    upload(client, profile="TAL")
    assert client.get("/profiles").json() == ["Tal"]
    assert len(client.get("/sessions", params={"profile": "Tal"}).json()) == 1
