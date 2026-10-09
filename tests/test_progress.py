"""Tests for pipeline.progress, on made-up reps."""

import dataclasses
import math

import numpy as np
import pytest

from pipeline.models import Fault, Rep, SessionResult, VideoInfo
from pipeline.progress import fatigue_index, session_progress

VIDEO = VideoInfo(path="v.mp4", fps=30.0, width=640, height=360, n_frames=900)
REP = Rep(1, 0, 30, 60, 85.0, 4.0, 170.0, 2.0, 1.0, 0.0, 0.0, 28, 30, 59, 40, 50)
DEPTH = "min_upper_arm_angle"


def reps_with_depths(*depths: float) -> list[Rep]:
    return [
        dataclasses.replace(REP, index=i, min_upper_arm_angle=d)
        for i, d in enumerate(depths, start=1)
    ]


def fault(rep: int, rule: str = "shallow", suppressed: bool = False) -> Fault:
    return Fault(rep, rule, "m", 20.0, (0, 60), 30, suppressed)


class TestFatigue:
    def test_reps_getting_shallower_is_positive(self):
        assert fatigue_index(np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])) == pytest.approx(8.0)

    def test_fewer_than_three_reps_has_no_thirds(self):
        assert math.isnan(fatigue_index(np.array([2.0, 9.0])))

    def test_unmeasured_reps_are_skipped_within_a_third(self):
        values = np.array([2.0, math.nan, 5.0, 5.0, math.nan, 10.0])
        assert fatigue_index(values) == pytest.approx(8.0)

    def test_a_third_with_nothing_measured_has_no_value(self):
        assert math.isnan(fatigue_index(np.array([math.nan, 5.0, 10.0])))


def test_a_session_counts_reps_with_a_shown_fault():
    """Two faults on rep 1 count once; a suppressed fault is not shown, so rep 2 is clean."""
    result = SessionResult(
        VIDEO,
        reps_with_depths(4.0, 6.0, 8.0, 30.0),
        [fault(1), fault(1, "hip_sag"), fault(2, suppressed=True), fault(4)],
        [],
    )
    progress = session_progress(result, [DEPTH])
    assert (progress.reps, progress.to_fix, progress.clean_share) == (4, 2, 0.5)


def test_depth_numbers_skip_unmeasured_reps():
    result = SessionResult(VIDEO, reps_with_depths(4.0, math.nan, 8.0, 12.0), [], [])
    progress = session_progress(result, [DEPTH])
    assert progress.average[DEPTH] == pytest.approx(8.0)
    assert progress.middle_half[DEPTH] == pytest.approx((6.0, 10.0))


def test_nothing_measured_gives_nan_not_zero():
    result = SessionResult(VIDEO, reps_with_depths(math.nan, math.nan), [], [])
    progress = session_progress(result, [DEPTH])
    assert math.isnan(progress.average[DEPTH])
    assert all(math.isnan(v) for v in progress.middle_half[DEPTH])
