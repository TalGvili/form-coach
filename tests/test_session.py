"""Tests for pipeline.session. analyze() itself needs a video, which CI doesn't have; it is
four calls to stages tested elsewhere. The JSON conversion is tested here."""

import json
import math

from pipeline.models import Fault, Rep, SessionResult, Unevaluated, VideoInfo
from pipeline.session import to_json

REP = Rep(
    index=1,
    start_frame=10,
    bottom_frame=40,
    end_frame=70,
    min_elbow_angle=85.0,
    min_upper_arm_angle=20.0,
    max_elbow_angle=math.nan,  # wrist outside the frame
    max_hip_drop=3.0,
    max_hip_rise=0.0,
    hip_sag_duration_s=0.0,
    hip_pike_duration_s=0.0,
    min_elbow_angle_frame=40,
    min_upper_arm_angle_frame=40,
    max_elbow_angle_frame=68,
    max_hip_drop_frame=35,
    max_hip_rise_frame=12,
)
RESULT = SessionResult(
    video=VideoInfo(path="clip.mp4", fps=30.0, width=1920, height=1080, n_frames=90),
    reps=[REP],
    faults=[Fault(1, "shallow", "Not reaching full depth", 20.0, (10, 70), frame=40)],
    unevaluated=[Unevaluated(1, "no_lockout")],
)


def test_nan_becomes_null_so_the_json_is_valid():
    data = to_json(RESULT)
    assert data["reps"][0]["max_elbow_angle"] is None
    text = json.dumps(data, allow_nan=False)  # raises if any NaN slipped through
    assert json.loads(text)["reps"][0]["max_elbow_angle"] is None


def test_every_part_of_the_result_is_kept():
    data = to_json(RESULT)
    assert data["video"]["fps"] == 30.0
    assert data["faults"][0] == {
        "rep_index": 1,
        "rule": "shallow",
        "message": "Not reaching full depth",
        "value": 20.0,
        "frames": [10, 70],
        "frame": 40,
        "suppressed": False,
    }
    assert data["unevaluated"] == [{"rep_index": 1, "rule": "no_lockout"}]
