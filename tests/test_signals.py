"""Tests for pipeline.signals, on synthetic points where the answer is known exactly.

No video files: they are not in git, and a test that needs one cannot run in CI.
"""

import numpy as np
import pytest

from pipeline import signals


def make_landmarks(points: dict[int, tuple[float, float]], n_frames: int = 1) -> np.ndarray:
    """A (n_frames, 33, 4) array with only the given landmarks set, the rest NaN."""
    landmarks = np.full((n_frames, 33, 4), np.nan)
    for index, (x, y) in points.items():
        landmarks[:, index, 0] = x
        landmarks[:, index, 1] = y
        landmarks[:, index, 3] = 1.0  # visibility
    return landmarks


class TestAngle:
    def test_right_angle_is_90(self):
        a, b, c = np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
        assert signals.angle(a, b, c) == pytest.approx(90.0)

    def test_straight_line_is_180(self):
        a, b, c = np.array([-1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
        assert signals.angle(a, b, c) == pytest.approx(180.0)

    def test_equilateral_triangle_is_60(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.5, np.sqrt(3) / 2])
        assert signals.angle(a, b, c) == pytest.approx(60.0)

    def test_vectorises_over_frames(self):
        a = np.array([[0.0, 1.0], [-1.0, 0.0]])
        b = np.array([[0.0, 0.0], [0.0, 0.0]])
        c = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert signals.angle(a, b, c) == pytest.approx([90.0, 180.0])


class TestUpperArmAngle:
    def test_horizontal_upper_arm_is_zero(self):
        # elbow level with the shoulder: upper arm parallel to the floor
        landmarks = make_landmarks({11: (100.0, 200.0), 13: (180.0, 200.0)})
        assert signals.upper_arm_angle_series(landmarks, signals.LEFT)[0] == pytest.approx(0.0)

    def test_vertical_upper_arm_is_90(self):
        landmarks = make_landmarks({11: (100.0, 200.0), 13: (100.0, 320.0)})
        assert signals.upper_arm_angle_series(landmarks, signals.LEFT)[0] == pytest.approx(90.0)

    def test_independent_of_facing(self):
        right = make_landmarks({11: (100.0, 200.0), 13: (180.0, 260.0)})
        left = make_landmarks({11: (100.0, 200.0), 13: (20.0, 260.0)})
        assert signals.upper_arm_angle_series(right, signals.LEFT)[0] == pytest.approx(
            signals.upper_arm_angle_series(left, signals.LEFT)[0]
        )


class TestElbowAngle:
    """Checks the landmark order: the angle must sit at the elbow, between shoulder and wrist."""

    def test_straight_arm_is_180(self):
        landmarks = make_landmarks({11: (100.0, 100.0), 13: (100.0, 200.0), 15: (100.0, 300.0)})
        assert signals.elbow_angle_series(landmarks, signals.LEFT)[0] == pytest.approx(180.0)

    def test_bent_arm_is_90(self):
        landmarks = make_landmarks({11: (100.0, 100.0), 13: (100.0, 200.0), 15: (200.0, 200.0)})
        assert signals.elbow_angle_series(landmarks, signals.LEFT)[0] == pytest.approx(90.0)


class TestTorsoTilt:
    def test_horizontal_body_is_zero(self):
        landmarks = make_landmarks({11: (100.0, 500.0), 27: (900.0, 500.0)})
        assert signals.torso_tilt_series(landmarks, signals.LEFT)[0] == pytest.approx(0.0)

    def test_standing_is_90(self):
        landmarks = make_landmarks({11: (500.0, 100.0), 27: (500.0, 900.0)})
        assert signals.torso_tilt_series(landmarks, signals.LEFT)[0] == pytest.approx(90.0)

    def test_independent_of_facing(self):
        head_left = make_landmarks({11: (100.0, 450.0), 27: (900.0, 500.0)})
        head_right = make_landmarks({11: (900.0, 450.0), 27: (100.0, 500.0)})
        assert signals.torso_tilt_series(head_left, signals.LEFT)[0] == pytest.approx(
            signals.torso_tilt_series(head_right, signals.LEFT)[0]
        )


class TestRuns:
    def test_finds_each_run_with_an_exclusive_stop(self):
        mask = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        assert signals.runs(mask) == [(1, 4), (6, 8)]

    def test_runs_touching_both_ends(self):
        # the padding is what lets a run start at 0 or end at len(mask)
        mask = np.array([1, 1, 0, 1], dtype=bool)
        assert signals.runs(mask) == [(0, 2), (3, 4)]

    def test_all_true_and_all_false(self):
        assert signals.runs(np.ones(5, dtype=bool)) == [(0, 5)]
        assert signals.runs(np.zeros(5, dtype=bool)) == []


class TestHipDeviation:
    """The sign convention decides hip_sag from hip_pike, so it gets the most tests."""

    STRAIGHT = {11: (0.0, 0.0), 23: (50.0, 0.0), 27: (100.0, 0.0)}
    SAGGING = {11: (0.0, 0.0), 23: (50.0, 10.0), 27: (100.0, 0.0)}  # y grows downward
    PIKING = {11: (0.0, 0.0), 23: (50.0, -10.0), 27: (100.0, 0.0)}

    def test_straight_body_is_zero(self):
        deviation = signals.hip_deviation_series(make_landmarks(self.STRAIGHT), signals.LEFT)
        assert deviation[0] == pytest.approx(0.0, abs=1e-9)

    def test_hip_below_the_line_is_positive(self):
        deviation = signals.hip_deviation_series(make_landmarks(self.SAGGING), signals.LEFT)
        assert deviation[0] > 0

    def test_hip_above_the_line_is_negative(self):
        deviation = signals.hip_deviation_series(make_landmarks(self.PIKING), signals.LEFT)
        assert deviation[0] < 0

    def test_sag_and_pike_have_equal_magnitude(self):
        sag = signals.hip_deviation_series(make_landmarks(self.SAGGING), signals.LEFT)[0]
        pike = signals.hip_deviation_series(make_landmarks(self.PIKING), signals.LEFT)[0]
        assert sag == pytest.approx(-pike)

    @pytest.mark.parametrize("points", [SAGGING, PIKING])
    def test_sign_survives_mirroring(self, points):
        """Swapping shoulder and ankle must not change the result.

        Four clips are filmed from the opposite side. A 2D cross product would flip sign
        here, because its sign depends on the direction of the shoulder-ankle line.
        """
        mirrored = {11: points[27], 23: points[23], 27: points[11]}
        original = signals.hip_deviation_series(make_landmarks(points), signals.LEFT)[0]
        flipped = signals.hip_deviation_series(make_landmarks(mirrored), signals.LEFT)[0]
        assert original == pytest.approx(flipped)

    def test_vertical_body_is_nan(self):
        vertical = {11: (50.0, 0.0), 23: (60.0, 50.0), 27: (50.0, 100.0)}
        deviation = signals.hip_deviation_series(make_landmarks(vertical), signals.LEFT)
        assert np.isnan(deviation[0])


class TestPickSide:
    def test_picks_the_more_visible_side(self):
        landmarks = np.full((10, 33, 4), np.nan)
        landmarks[:, signals.LEFT.indices, 3] = 0.3
        landmarks[:, signals.RIGHT.indices, 3] = 0.9
        assert signals.pick_side(landmarks) is signals.RIGHT

        landmarks[:, signals.LEFT.indices, 3] = 0.95
        assert signals.pick_side(landmarks) is signals.LEFT


class TestInPositionWindow:
    def test_picks_the_longest_horizontal_run(self):
        # a brief horizontal moment during setup, then the real push-up stretch
        tilt = np.concatenate(
            [
                np.full(20, 80.0),  # standing
                np.full(5, 10.0),  # a stray horizontal moment
                np.full(20, 80.0),  # standing again
                np.full(100, 10.0),  # the actual push-ups
                np.full(15, 80.0),  # getting up
            ]
        )
        assert signals.in_position_window(tilt, max_tilt_deg=35.0) == (45, 145)

    def test_nan_counts_as_not_in_position(self):
        tilt = np.concatenate([np.full(10, np.nan), np.full(30, 5.0)])
        assert signals.in_position_window(tilt, max_tilt_deg=35.0) == (10, 40)

    def test_no_horizontal_frames_returns_an_empty_window(self):
        # standing the whole time: there is no push-up to search, not the whole clip
        assert signals.in_position_window(np.full(50, 80.0), max_tilt_deg=35.0) == (0, 0)


class TestInterpolateGaps:
    def test_fills_a_short_gap(self):
        signal = np.array([0.0, np.nan, np.nan, 3.0])
        assert signals.interpolate_gaps(signal, max_gap=3) == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_leaves_a_long_gap_alone(self):
        signal = np.array([0.0, np.nan, np.nan, np.nan, 4.0])
        assert np.isnan(signals.interpolate_gaps(signal, max_gap=2)[1:4]).all()

    def test_leaves_leading_and_trailing_gaps_alone(self):
        signal = np.array([np.nan, 1.0, 2.0, np.nan])
        filled = signals.interpolate_gaps(signal, max_gap=5)
        assert np.isnan(filled[0]) and np.isnan(filled[-1])

    def test_signal_without_gaps_is_unchanged(self):
        signal = np.array([1.0, 2.0, 3.0])
        assert signals.interpolate_gaps(signal, max_gap=5) == pytest.approx(signal)


class TestSmooth:
    def test_reduces_noise_but_keeps_the_wave(self):
        t = np.linspace(0, 16, 480)  # 16 s at 30 fps
        clean = 45 * np.cos(2 * np.pi * t / 2)
        noisy = clean + np.random.default_rng(0).normal(0, 3, t.size)
        smoothed = signals.smooth(noisy, fps=30.0, window_s=0.4)
        assert np.abs(smoothed - clean).mean() < np.abs(noisy - clean).mean()

    def test_nan_does_not_spread_into_neighbours(self):
        signal = np.concatenate([np.ones(60), [np.nan], np.ones(60) * 2])
        smoothed = signals.smooth(signal, fps=30.0, window_s=0.4)
        assert np.isnan(smoothed).sum() == 1
