"""Rep segmentation: find each rep in the depth signal and measure it.

A rep runs top -> bottom -> top. Bottoms are the dips of the smoothed upper-arm angle, found
with scipy's find_peaks; the top between two bottoms is where that angle is highest. Each
rep then gets one number per field of pipeline.models.Rep.

The first and last rep have a neighbouring rep on one side only. Their open side is cut at
the clip's own typical descent or ascent length: left to run to the edge of the in-position
window, it takes in getting down and getting up, which reads as a large hip pike.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import find_peaks

from pipeline import signals
from pipeline.errors import AnalysisError, NoPersonError, SingleRepError
from pipeline.landmarks import load_or_extract
from pipeline.models import Rep, VideoInfo


@dataclass(frozen=True)
class RepConfig:
    """Segmentation settings: the `segmentation` section of the config file."""

    min_detected_fraction: float
    max_tilt_deg: float
    max_gap_frames: int
    smoothing_window_s: float
    min_prominence_fraction: float
    min_rep_spacing_s: float
    min_range_deg: float
    min_return_fraction: float
    hip_moving_fraction: float
    sag_threshold_deg: float
    pike_threshold_deg: float


def load_config(path: Path) -> RepConfig:
    """Read the segmentation settings. A missing or misspelled key raises a TypeError."""
    with path.open(encoding="utf-8") as handle:
        return RepConfig(**yaml.safe_load(handle)["segmentation"])


@dataclass(frozen=True)
class ClipSignals:
    """The per-frame series one clip's reps are measured from, indexed by frame in the whole
    clip and NaN outside the in-position window.

    The *_seen arrays say, per frame, whether every landmark a series needs is inside the
    video frame. MediaPipe still reports a landmark beyond the frame edge by extrapolating
    from the body, so a value computed from it is a guess.
    """

    depth: np.ndarray  # upper-arm angle: 0 = parallel to the floor
    depth_raw: np.ndarray  # the same, gap-filled but not smoothed: for reading the bottom
    elbow: np.ndarray  # elbow angle: 180 = straight arm
    hip: np.ndarray  # hip deviation: + below the body line, - above it
    depth_seen: np.ndarray
    elbow_seen: np.ndarray
    hip_seen: np.ndarray


def find_rep_bottoms(
    depth: np.ndarray,
    fps: float,
    min_prominence_fraction: float,
    min_rep_spacing_s: float,
    min_range_deg: float,
) -> np.ndarray:
    """Frame indices where the depth signal dips, one per rep.

    find_peaks finds maxima, so the signal is negated. The required prominence is a share of
    the signal's own range (5th to 95th percentile), so it does not depend on camera distance
    or on how deep this person goes. Unknown frames take the highest value, so they can
    never be mistaken for a bottom.

    A share of nothing is nothing: when the arm barely moves all clip, the range is only
    jitter and so is every dip. Below min_range_deg the clip has no reps.
    """
    if np.isnan(depth).all():
        return np.array([], dtype=int)
    filled = np.where(np.isnan(depth), np.nanmax(depth), depth)
    spread = np.percentile(filled, 95) - np.percentile(filled, 5)
    if spread < min_range_deg:
        return np.array([], dtype=int)
    bottoms, _ = find_peaks(
        -filled,
        prominence=spread * min_prominence_fraction,
        distance=max(1, int(fps * min_rep_spacing_s)),
    )
    return bottoms


def rep_windows(
    depth: np.ndarray, bottoms: np.ndarray, window: tuple[int, int]
) -> list[tuple[int, int]]:
    """(first_frame, last_frame) of each rep, both inclusive: top -> bottom -> top.

    Between two bottoms the top is where the depth signal peaks, and it is shared: it ends one
    rep and starts the next. The first rep has no top before it and the last none after it,
    so those sides are cut at this clip's typical descent / ascent length (median over the
    gaps between its bottoms), and never leave the in-position window [start, stop).
    A single bottom has no gap to learn that length from, so it raises SingleRepError.
    """
    if len(bottoms) < 2:
        raise SingleRepError("Only one rep found. Film at least two reps to be analysed.")
    start, stop = window
    pairs = list(zip(bottoms[:-1], bottoms[1:], strict=True))
    tops = [int(a + np.nanargmax(depth[a:b])) for a, b in pairs]
    descent = np.median([b - t for t, (_, b) in zip(tops, pairs, strict=True)])
    ascent = np.median([t - a for t, (a, _) in zip(tops, pairs, strict=True)])
    first = max(start, int(bottoms[0] - descent))
    last = min(stop - 1, int(bottoms[-1] + ascent))
    return list(zip([first, *tops], [*tops, last], strict=True))


def completed_reps(
    depth: np.ndarray,
    bottoms: np.ndarray,
    windows: list[tuple[int, int]],
    min_return_fraction: float,
) -> np.ndarray:
    """Whether each rep climbs back up. A rep only counts once the person returns to the top.

    Rise = the highest depth value after the bottom minus the value at the bottom, compared
    with this clip's median rise. A no-lockout rep still rises most of the way; a failed rep,
    where the person never pushes back up, barely rises at all.
    """
    rises = np.array(
        [
            np.nanmax(depth[b : last + 1]) - depth[b]
            for b, (_, last) in zip(bottoms, windows, strict=True)
        ]
    )
    return rises >= min_return_fraction * np.median(rises)


def _in_frame(landmarks: np.ndarray, indices: list[int], info: VideoInfo) -> np.ndarray:
    """Per frame: all the given landmarks lie inside the video frame. NaN counts as outside."""
    x, y = landmarks[:, indices, 0], landmarks[:, indices, 1]
    inside = (x >= 0) & (x <= info.width) & (y >= 0) & (y <= info.height)
    return inside.all(axis=1)


def clip_signals(
    landmarks: np.ndarray,
    info: VideoInfo,
    side: signals.Side,
    window: tuple[int, int],
    cfg: RepConfig,
) -> ClipSignals:
    """Compute, gap-fill and smooth every series the reps are measured from.

    Frames outside the in-position window are blanked to NaN first. smooth() works on each
    run of valid frames separately, so the push-ups are smoothed on their own. Smoothed
    across the edge, the jump to standing makes the fitted curve overshoot and invent a dip.

    Depth is also kept unsmoothed: smoothing finds the bottom reliably but rounds off its
    tip, reading a narrow dip several degrees too high.
    """
    start, stop = window

    def fill(series: np.ndarray) -> np.ndarray:
        inside = np.full_like(series, np.nan)
        inside[start:stop] = series[start:stop]
        return signals.interpolate_gaps(inside, cfg.max_gap_frames)

    def prepare(series: np.ndarray) -> np.ndarray:
        return signals.smooth(fill(series), info.fps, cfg.smoothing_window_s)

    upper_arm = signals.upper_arm_angle_series(landmarks, side)
    return ClipSignals(
        depth=prepare(upper_arm),
        depth_raw=fill(upper_arm),
        elbow=prepare(signals.elbow_angle_series(landmarks, side)),
        hip=prepare(signals.hip_deviation_series(landmarks, side)),
        depth_seen=_in_frame(landmarks, [side.shoulder, side.elbow], info),
        elbow_seen=_in_frame(landmarks, [side.shoulder, side.elbow, side.wrist], info),
        hip_seen=_in_frame(landmarks, [side.shoulder, side.hip, side.ankle], info),
    )


def _extreme(
    series: np.ndarray, window: slice, find: Callable[[np.ndarray], np.intp]
) -> tuple[float, int]:
    """(value, frame) of the extreme of series inside window; find is np.nanargmin/argmax.
    An all-NaN window has no extreme: (NaN, window.start)."""
    if np.isnan(series[window]).all():
        return float("nan"), window.start
    frame = window.start + int(find(series[window]))
    return float(series[frame]), frame


def _seen_or_nan(value: float, frame: int, seen: np.ndarray) -> float:
    """The value if the frame it came from was observed, otherwise NaN: "could not measure"."""
    return value if seen[frame] else float("nan")


def _longest_run_s(mask: np.ndarray, fps: float) -> float:
    """Length in seconds of the longest continuous stretch of True."""
    return max((stop - start for start, stop in signals.runs(mask)), default=0) / fps


def measure_rep(
    index: int, bottom: int, window: tuple[int, int], s: ClipSignals, fps: float, cfg: RepConfig
) -> Rep:
    """Turn one rep's frames into the numbers stored on a Rep."""
    first, last = window
    whole = slice(first, last + 1)
    # Lockout is judged at the top this rep rises TO. The top it started from belongs to the
    # previous rep; including it would let one locked rep hide the next rep's bent arms.
    rising = slice(bottom, last + 1)

    # Hip faults are judged while the arms move, not at the top: there the person may still
    # be settling into the plank, pausing, or already getting up, and the hip moves freely.
    low = s.depth[bottom]
    reach = low + cfg.hip_moving_fraction * (np.nanmax(s.depth[whole]) - low)
    hip = np.full_like(s.hip, np.nan)
    hip[whole] = np.where(s.depth[whole] <= reach, s.hip[whole], np.nan)

    low_elbow, low_elbow_at = _extreme(s.elbow, whole, np.nanargmin)
    high_elbow, high_elbow_at = _extreme(s.elbow, rising, np.nanargmax)
    drop, drop_at = _extreme(hip, whole, np.nanargmax)
    rise, rise_at = _extreme(hip, whole, np.nanargmin)

    max_hip_drop = _seen_or_nan(max(drop, 0.0), drop_at, s.hip_seen)
    max_hip_rise = _seen_or_nan(max(-rise, 0.0), rise_at, s.hip_seen)
    sag_s = _longest_run_s(hip[whole] > cfg.sag_threshold_deg, fps)
    pike_s = _longest_run_s(hip[whole] < -cfg.pike_threshold_deg, fps)

    return Rep(
        index=index,
        start_frame=first,
        bottom_frame=bottom,
        end_frame=last,
        min_elbow_angle=_seen_or_nan(low_elbow, low_elbow_at, s.elbow_seen),
        # Found on the smoothed signal, read from the raw one: smoothing rounds off the tip
        # of a narrow dip (clip16 rep 1: raw 2.3 deg, smoothed 6.5). EXPERIMENTS.md, no. 8.
        min_upper_arm_angle=_seen_or_nan(float(s.depth_raw[bottom]), bottom, s.depth_seen),
        max_elbow_angle=_seen_or_nan(high_elbow, high_elbow_at, s.elbow_seen),
        max_hip_drop=max_hip_drop,
        max_hip_rise=max_hip_rise,
        hip_sag_duration_s=float("nan") if np.isnan(max_hip_drop) else sag_s,
        hip_pike_duration_s=float("nan") if np.isnan(max_hip_rise) else pike_s,
        min_elbow_angle_frame=low_elbow_at,
        min_upper_arm_angle_frame=bottom,
        max_elbow_angle_frame=high_elbow_at,
        max_hip_drop_frame=drop_at,
        max_hip_rise_frame=rise_at,
    )


def segment_reps(landmarks: np.ndarray, info: VideoInfo, cfg: RepConfig) -> list[Rep]:
    """Find and measure every completed rep in one clip. Reps are numbered from 1."""
    detected = ~np.isnan(landmarks).all(axis=(1, 2))  # frames with a person in them
    if detected.size == 0 or detected.mean() < cfg.min_detected_fraction:
        raise NoPersonError(
            "Couldn't find a person in the video. Check that your whole body is in the frame."
        )
    side = signals.pick_side(landmarks)
    tilt = signals.torso_tilt_series(landmarks, side)
    start, stop = signals.in_position_window(tilt, cfg.max_tilt_deg)
    s = clip_signals(landmarks, info, side, (start, stop), cfg)

    # search only inside the window, then shift back to frame numbers in the whole clip
    bottoms = start + find_rep_bottoms(
        s.depth[start:stop],
        info.fps,
        cfg.min_prominence_fraction,
        cfg.min_rep_spacing_s,
        cfg.min_range_deg,
    )
    if len(bottoms) == 0:
        return []
    # A failed rep is not a rep, so it must not decide where its neighbour ends: the top
    # between them may be the person getting up. Drop it and recompute until all complete.
    while True:
        windows = rep_windows(s.depth, bottoms, (start, stop))
        completed = completed_reps(s.depth, bottoms, windows, cfg.min_return_fraction)
        if completed.all():
            break
        bottoms = bottoms[completed]

    return [
        measure_rep(k, int(bottom), window, s, info.fps, cfg)
        for k, (bottom, window) in enumerate(zip(bottoms, windows, strict=True), start=1)
    ]


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2):
        print("usage: python -m pipeline.reps <video> [config]", file=sys.stderr)
        return 2
    config = Path(argv[1] if len(argv) == 2 else "configs/pushup.yaml")
    info, landmarks = load_or_extract(argv[0])
    try:
        reps = segment_reps(landmarks, info, load_config(config))
    except AnalysisError as error:
        print(f"{info.path}: {error}", file=sys.stderr)
        return 1
    print(f"{info.path}: {len(reps)} reps")
    header = ("rep", "frames", "depth", "lockout", "hip drop", "hip rise")
    print("  {:>3} {:>11} {:>6} {:>8} {:>9} {:>9}".format(*header))
    for r in reps:
        print(
            f"  {r.index:>3} {r.start_frame:>5}-{r.end_frame:<5} {r.min_upper_arm_angle:>6.1f}"
            f" {r.max_elbow_angle:>8.1f} {r.max_hip_drop:>9.1f} {r.max_hip_rise:>9.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
