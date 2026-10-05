"""Tests for app.main, the HTTP layer only.

analyze, render and probe_video are replaced by fakes, so no model, video or MediaPipe is
needed. What is tested is what this layer owns: status codes, messages, and the response
shape. The text-file test uses the real probe_video, since refusing a non-video is its job.
"""

from pathlib import Path

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
    monkeypatch.setattr(main, "probe_video", lambda path: INFO)
    monkeypatch.setattr(main, "analyze", lambda path, config: SessionResult(INFO, [REP], [], []))
    monkeypatch.setattr(main, "render", lambda result, titles, out: out.write_bytes(b"video"))
    return TestClient(main.app)


def upload(client: TestClient, content: bytes = b"any bytes"):
    return client.post("/analyze", files={"video": ("set.mp4", content, "video/mp4")})


def test_a_video_returns_the_report_and_a_link_to_the_annotated_video(client: TestClient):
    response = upload(client)
    assert response.status_code == 200
    report = response.json()
    assert len(report["reps"]) == 1
    assert report["reps"][0]["min_elbow_angle"] is None  # NaN became null: valid JSON
    assert "path" not in report["video"]  # the server's file layout stays private
    assert report["hint"] is None
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
