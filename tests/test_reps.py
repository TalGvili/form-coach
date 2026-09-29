"""Tests for pipeline.reps, on synthetic clips where every rep is known exactly.

synthetic_clip builds a side-view landmark array frame by frame: standing, a plank, a
number of push-ups, a plank, standing. Each push-up's depth, how straight the arm gets at
the top, and how far the hip sags are all chosen by the test.
"""

import numpy as np
import pytest

from pipeline import signals
from pipeline.models import VideoInfo
from pipeline.reps import (
    RepConfig,
    SingleRepError,
    find_rep_bottoms,
    load_config,
    segment_reps,
)

FPS = 30.0
PERIOD = int(2 * FPS)  # one push-up every two seconds
INFO = VideoInfo(path="synthetic", fps=FPS, width=1920, height=1080, n_frames=0)
CFG = RepConfig(
    max_tilt_deg=35,
    max_gap_frames=5,
    smoothing_window_s=0.4,
    min_prominence_fraction=0.3,
    min_rep_spacing_s=0.8,
    min_range_deg=10,
    min_return_fraction=0.45,
    hip_moving_fraction=0.9,
    hip_duration_threshold_deg=15,
)
L = signals.LEFT
STANDING = {
    L.shoulder: (1500, 200),
    L.elbow: (1500, 350),
    L.wrist: (1500, 500),
    L.hip: (1500, 550),
    L.ankle: (1510, 900),
}


def _pose(upper_arm_deg: float, bend_deg: float, sag_px: float) -> dict[int, np.ndarray]:
    """Plank landmarks: upper arm at the given angle, elbow bent by bend_deg, hip sagging."""
    shoulder, ankle = np.array([600.0, 500.0]), np.array([1400.0, 560.0])
    theta, bend = np.radians(upper_arm_deg), np.radians(bend_deg)
    upper = np.array([np.cos(theta), np.sin(theta)])  # y grows downward
    rotate = np.array([[np.cos(bend), -np.sin(bend)], [np.sin(bend), np.cos(bend)]])
    elbow = shoulder + 110 * upper
    wrist = elbow + 110 * rotate @ upper  # elbow angle = 180 - bend
    hip = (shoulder + ankle) / 2 + np.array([0.0, sag_px])
    return {L.shoulder: shoulder, L.elbow: elbow, L.wrist: wrist, L.hip: hip, L.ankle: ankle}


def synthetic_clip(
    top_bends: list[float],
    sags: list[float] | None = None,
    failed_last: bool = False,
    get_up: bool = False,
) -> np.ndarray:
    """(n_frames, 33, 4) landmarks for standing, len(top_bends) push-ups, standing.

    top_bends[k]: degrees short of straight the arm stays at the top of rep k.
    sags[k]: pixels the hip drops below the body line at the bottom of rep k.
    failed_last: then one more rep that comes only 40% of the way back up and collapses.
    get_up: the failed rep sinks back only part of the way and stays stuck there (clip11),
    then pushes off the floor to straight arms, with a dip on the way.
    """
    sags = sags or [0.0] * len(top_bends)
    poses: list[dict | None] = [None] * int(FPS)  # standing
    poses += [_pose(88, 0, 0)] * int(FPS / 2)  # holding the plank
    for k, (top_bend, sag) in enumerate(zip(top_bends, sags, strict=True)):
        previous_top = top_bends[k - 1] if k else 0.0
        for i in range(PERIOD):
            p = (1 - np.cos(2 * np.pi * i / PERIOD)) / 2  # 0 at the tops, 1 at the bottom
            top = previous_top if i < PERIOD / 2 else top_bend
            poses.append(_pose(88 - 84 * p, 90 * p + (1 - p) * top, sag * p))
    last_top = top_bends[-1]
    if failed_last:
        rest = 0.8 if get_up else 1.0  # stuck partway, or flat on the floor
        path = np.r_[
            (1 - np.cos(np.linspace(0, np.pi, PERIOD // 2))) / 2,  # all the way down
            np.linspace(1, 0.6, 12),  # a push that gets 40% of the way up
            np.linspace(0.6, rest, 12),  # and gives way
            np.full(int(FPS), rest),  # held there
        ]
        if get_up:
            path = np.r_[
                path,
                np.linspace(rest, 0, 15),  # pushing off the floor
                0.4 * (1 - np.cos(np.linspace(0, 2 * np.pi, 24))) / 2,  # a dip on the way up
            ]
        poses += [_pose(88 - 84 * p, 90 * p + (1 - p) * last_top, 0) for p in path]
    else:
        poses += [_pose(88, last_top, 0)] * int(FPS / 2)
    poses += [None] * int(FPS)  # standing

    landmarks = np.full((len(poses), 33, 4), np.nan)
    landmarks[:, signals.RIGHT.indices, 3] = 0.2  # the far side: barely visible
    for f, pose in enumerate(poses):
        for index, xy in (STANDING if pose is None else pose).items():
            landmarks[f, index, :2] = xy
            landmarks[f, index, 3] = 1.0
    return landmarks


def test_find_rep_bottoms_counts_eight_reps():
    t = np.linspace(0, 16, 480)  # 16 s at 30 fps
    signal = 125 + 45 * np.cos(2 * np.pi * t / 2)
    signal += np.random.default_rng(0).normal(0, 3, t.size)
    assert len(find_rep_bottoms(signal, 30.0, 0.3, 0.8, 10)) == 8


def test_find_rep_bottoms_on_a_flat_signal_finds_nothing():
    assert len(find_rep_bottoms(np.full(100, 50.0), 30.0, 0.3, 0.8, 10)) == 0


def test_find_rep_bottoms_ignores_jitter_when_nothing_moves():
    """A 2-degree wobble has dips too, and each spans far more than 30% of a 4-degree range."""
    t = np.linspace(0, 16, 480)
    wobble = 85 + 2 * np.sin(2 * np.pi * t / 1.5)
    assert len(find_rep_bottoms(wobble, 30.0, 0.3, 0.8, 10)) == 0
    assert len(find_rep_bottoms(wobble, 30.0, 0.3, 0.8, 0)) > 0  # what happened before


def test_finds_every_rep_and_ignores_standing():
    reps = segment_reps(synthetic_clip([0.0] * 6), INFO, CFG)
    assert [r.index for r in reps] == [1, 2, 3, 4, 5, 6]
    assert all(r.min_upper_arm_angle == pytest.approx(4, abs=2) for r in reps)


def test_reps_run_top_to_bottom_to_top_and_share_tops():
    reps = segment_reps(synthetic_clip([0.0] * 4), INFO, CFG)
    for rep in reps:
        assert rep.start_frame < rep.bottom_frame < rep.end_frame
    for before, after in zip(reps, reps[1:], strict=False):
        assert before.end_frame == after.start_frame


def test_first_and_last_rep_are_cut_at_a_typical_rep_length():
    reps = segment_reps(synthetic_clip([0.0] * 4), INFO, CFG)
    half = PERIOD // 2
    assert reps[0].bottom_frame - reps[0].start_frame == pytest.approx(half, abs=3)
    assert reps[-1].end_frame - reps[-1].bottom_frame == pytest.approx(half, abs=3)


def test_a_single_rep_is_refused_rather_than_guessed():
    """One rep has no neighbour to learn the clip's rhythm from, so where it starts and ends
    is unknown."""
    with pytest.raises(SingleRepError):
        segment_reps(synthetic_clip([0.0]), INFO, CFG)


def test_a_failed_last_rep_is_not_counted():
    reps = segment_reps(synthetic_clip([0.0] * 5, failed_last=True), INFO, CFG)
    assert len(reps) == 5


def test_getting_up_after_a_failed_rep_does_not_complete_it():
    """clip11: pushing off the floor makes a dip of its own. That dip is rejected, but while
    it stands, the top before it is the getting up, and the failed rep appears to rise to it."""
    reps = segment_reps(synthetic_clip([0.0] * 5, failed_last=True, get_up=True), INFO, CFG)
    assert len(reps) == 5


def test_lockout_is_judged_at_the_top_the_rep_rises_to():
    """Alternating straight and bent tops: each rep must report its own top, not the
    locked-out top of the rep before it."""
    reps = segment_reps(synthetic_clip([0.0, 40.0, 0.0, 40.0, 0.0, 40.0]), INFO, CFG)
    highest = [r.max_elbow_angle for r in reps]
    assert all(angle > 170 for angle in highest[0::2])
    assert all(angle < 150 for angle in highest[1::2])


def test_a_sagging_rep_has_a_positive_drop_and_a_duration():
    reps = segment_reps(synthetic_clip([0.0] * 5, sags=[0, 0, 80, 0, 0]), INFO, CFG)
    sagging = reps[2]
    assert sagging.max_hip_drop > CFG.hip_duration_threshold_deg
    assert 0.5 < sagging.hip_sag_duration_s < 1.1
    assert sagging.max_hip_rise == pytest.approx(0, abs=1)
    for rep in reps[:2] + reps[3:]:
        assert rep.max_hip_drop < 5
        assert rep.hip_sag_duration_s == 0


def test_the_hip_moving_while_the_arms_rest_at_the_top_is_ignored():
    """A pike held only at the top between reps 1 and 2 (arms straight, hips up) is a pause,
    not a fault in either rep."""
    landmarks = synthetic_clip([0.0] * 4)
    top = int(FPS) + int(FPS / 2) + PERIOD  # the shared top of reps 1 and 2
    landmarks[top - 3 : top + 4, L.hip, 1] -= 80  # hips raised, for 0.23 s
    reps = segment_reps(landmarks, INFO, CFG)
    assert reps[0].max_hip_rise < 5
    assert reps[1].max_hip_rise < 5


def test_a_piking_rep_has_a_positive_rise_and_a_duration():
    """The pike mirror of the sag test: a negative sag raises the hip above the body line."""
    reps = segment_reps(synthetic_clip([0.0] * 5, sags=[0, -80, 0, 0, 0]), INFO, CFG)
    piking = reps[1]
    assert piking.max_hip_rise > CFG.hip_duration_threshold_deg
    assert 0.5 < piking.hip_pike_duration_s < 1.1
    assert piking.max_hip_drop == pytest.approx(0, abs=1)
    for rep in reps[:1] + reps[2:]:
        assert rep.max_hip_rise < 5
        assert rep.hip_pike_duration_s == 0


def test_a_clip_without_push_ups_has_no_reps():
    """Standing, a plank held still, standing: nothing dips, so nothing is a rep."""
    landmarks = synthetic_clip([0.0] * 3)
    plank = landmarks[int(FPS) + 1]  # a frame from the first plank hold
    landmarks[int(FPS) : -int(FPS)] = plank
    assert segment_reps(landmarks, INFO, CFG) == []


def test_a_hip_never_detected_gives_nan_hip_metrics():
    """No hip values in a rep at all: "could not measure", not a crash."""
    landmarks = synthetic_clip([0.0] * 4)
    landmarks[:, L.hip, :2] = np.nan
    reps = segment_reps(landmarks, INFO, CFG)
    assert len(reps) == 4
    assert all(np.isnan(r.max_hip_drop) and np.isnan(r.hip_sag_duration_s) for r in reps)
    assert all(np.isnan(r.max_hip_rise) and np.isnan(r.hip_pike_duration_s) for r in reps)


def test_a_wrist_outside_the_frame_gives_nan_elbow_metrics():
    landmarks = synthetic_clip([0.0] * 4)
    landmarks[:, L.wrist, 1] += 1000  # below the bottom edge of a 1080-pixel frame
    reps = segment_reps(landmarks, INFO, CFG)
    assert len(reps) == 4
    assert all(np.isnan(r.max_elbow_angle) and np.isnan(r.min_elbow_angle) for r in reps)
    assert not any(np.isnan(r.min_upper_arm_angle) for r in reps)  # depth needs no wrist


def test_the_config_file_matches_repconfig():
    assert isinstance(load_config(), RepConfig)
