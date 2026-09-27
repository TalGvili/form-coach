"""Per-frame signals derived from landmarks: angles, body-line deviation, smoothing.

Everything here turns the (n_frames, 33, 4) landmark array into one number per frame.
Nothing here knows what a rep is, and nothing here decides whether anything is a fault.

Thresholds are parameters rather than defaults on purpose: they are the values tuned in
Phase 5, and they belong in configs/*.yaml rather than in this file.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

X, Y, VISIBILITY = 0, 1, 3


@dataclass(frozen=True)
class Side:
    """Landmark indices for one side of the body.

    A side view hides the far limb, so every measurement uses one side throughout.
    Picking it once and passing it around keeps that choice explicit.
    """

    name: str
    shoulder: int
    elbow: int
    wrist: int
    hip: int
    ankle: int

    @property
    def indices(self) -> list[int]:
        return [self.shoulder, self.elbow, self.wrist, self.hip, self.ankle]


LEFT = Side("left", shoulder=11, elbow=13, wrist=15, hip=23, ankle=27)
RIGHT = Side("right", shoulder=12, elbow=14, wrist=16, hip=24, ankle=28)


def pick_side(landmarks: np.ndarray) -> Side:
    """The side facing the camera, by mean visibility.

    MediaPipe reports a position for the far limb too, but it is inferred from the body
    rather than seen, so its visibility score is low.
    """
    left = np.nanmean(landmarks[:, LEFT.indices, VISIBILITY])
    right = np.nanmean(landmarks[:, RIGHT.indices, VISIBILITY])
    return LEFT if left >= right else RIGHT


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b, in degrees, from the dot product.

    Works on single points of shape (2,) returning a scalar, or on (n, 2) arrays
    returning one angle per frame. `np.clip` before `arccos` because floating-point
    error can push the cosine a hair outside [-1, 1], which would produce NaN.
    """
    ba, bc = a - b, c - b
    norms = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (ba * bc).sum(axis=-1) / norms
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def elbow_angle_series(landmarks: np.ndarray, side: Side) -> np.ndarray:
    """Shoulder-elbow-wrist angle per frame. ~180 with the arm straight.

    Needs the wrist, which is the landmark most likely to fall outside the frame when
    the floor is below the bottom edge. Prefer upper_arm_angle_series for depth.
    """
    return angle(
        landmarks[:, side.shoulder, :2],
        landmarks[:, side.elbow, :2],
        landmarks[:, side.wrist, :2],
    )


def upper_arm_angle_series(landmarks: np.ndarray, side: Side) -> np.ndarray:
    """Angle of the upper arm above horizontal. 0 = parallel to the floor.

    This is both the depth metric and the segmentation signal: it needs only shoulder
    and elbow, and being an angle it is unchanged by the body drifting around the frame.
    Absolute values on both components make it independent of which way the person faces.
    """
    delta = landmarks[:, side.elbow, :2] - landmarks[:, side.shoulder, :2]
    return np.degrees(np.arctan2(np.abs(delta[:, Y]), np.abs(delta[:, X])))


def torso_tilt_series(landmarks: np.ndarray, side: Side) -> np.ndarray:
    """Angle of the shoulder-ankle line from horizontal. ~0 lying down, ~90 standing."""
    delta = landmarks[:, side.ankle, :2] - landmarks[:, side.shoulder, :2]
    return np.degrees(np.arctan2(np.abs(delta[:, Y]), np.abs(delta[:, X])))


def hip_deviation_series(landmarks: np.ndarray, side: Side) -> np.ndarray:
    """Signed degrees the hip sits off the shoulder-ankle line.

    Positive means the hip is BELOW the line (sagging), negative means ABOVE it (piking).
    The magnitude alone cannot tell the two apart, since both bend the body the same
    amount; only the sign distinguishes them, and it decides two of the four faults.

    The sign compares the hip's y against the line's y at the hip's x. That is a fact
    about vertical position, so it holds whichever way the person faces. A 2D cross
    product would be the natural alternative, but its sign depends on the direction of
    the line and so flips on a mirrored clip.
    """
    shoulder = landmarks[:, side.shoulder, :2]
    hip = landmarks[:, side.hip, :2]
    ankle = landmarks[:, side.ankle, :2]

    magnitude = 180.0 - angle(shoulder, hip, ankle)

    run = ankle[:, X] - shoulder[:, X]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (hip[:, X] - shoulder[:, X]) / run
        line_y = shoulder[:, Y] + t * (ankle[:, Y] - shoulder[:, Y])

    # image y grows downward, so the hip being below the line means a larger y
    signed = np.where(hip[:, Y] > line_y, magnitude, -magnitude)
    # a near-vertical body has no meaningful line height at the hip's x
    return np.where(np.abs(run) < 1e-6, np.nan, signed)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Each continuous run of True in a boolean array, as (start, stop), stop exclusive.

    np.diff marks +1 where a run starts and -1 where it ends. Padding with a False at
    both ends makes a run that touches either end of the array still produce both
    edges. The edges always alternate start, stop, start, stop.
    """
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2], strict=True)]


def in_position_window(tilt: np.ndarray, max_tilt_deg: float) -> tuple[int, int]:
    """The longest stretch where the body is roughly horizontal, as (start, stop).

    Clips open and close with footage that is not reps: walking in, kneeling down,
    getting up. While standing, the upper-arm angle sits mid-range, so peak finding can
    count the setup as a rep. Taking the longest run rather than every qualifying frame
    stops a stray horizontal moment during setup from reopening the window.
    """
    # unknown tilt counts as standing, so frames without data are excluded
    horizontal = np.nan_to_num(tilt, nan=90.0) < max_tilt_deg
    runs = _runs(horizontal)
    if not runs:
        return 0, len(tilt)
    return max(runs, key=lambda run: run[1] - run[0])


def interpolate_gaps(signal: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill runs of NaN up to max_gap frames long; leave longer runs alone.

    A few missing frames are a detection hiccup and interpolating across them is
    honest. A long run means the person left the frame, and inventing values there
    would be fabrication.
    """
    filled = signal.copy()
    missing = np.isnan(signal)
    known = np.flatnonzero(~missing)
    for start, stop in _runs(missing):
        # only interior gaps can be interpolated; leading and trailing runs have no
        # value on one side to interpolate from
        if stop - start > max_gap or start == 0 or stop == len(signal):
            continue
        span = np.arange(start, stop)
        filled[span] = np.interp(span, known, signal[known])
    return filled


def smooth(signal: np.ndarray, fps: float, window_s: float, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing, applied to each run of valid values separately.

    Savgol keeps the shape of the dips better than a moving average, which matters
    because the dips are the reps. Running it per valid run stops a single NaN from
    spreading across a whole window. Runs shorter than the window are left as they are.
    """
    window = int(fps * window_s) | 1  # forced odd, as savgol_filter requires
    out = signal.copy()
    for start, stop in _runs(~np.isnan(signal)):
        if stop - start > window:
            out[start:stop] = savgol_filter(signal[start:stop], window, polyorder)
    return out
