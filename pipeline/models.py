"""Core dataclasses passed between pipeline stages: VideoInfo, Rep, Fault, SessionResult."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int


@dataclass(frozen=True)
class Rep:
    """One completed rep and its measurements.

    All angles are in degrees; `_s` marks a value in seconds. Frame indices are
    absolute positions in the source video. Field names are the `metric:` values the
    YAML rules look up, so renaming one here means renaming it in the config too.
    """

    index: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    min_elbow_angle: float  # depth (shallow)
    min_upper_arm_angle: float  # 0 = upper arm parallel to floor (shallow)
    max_elbow_angle: float  # lockout at the top (no_lockout)
    max_hip_drop: float  # hip below the body line (hip_sag)
    max_hip_rise: float  # hip above the body line (hip_pike)
    hip_sag_duration_s: float
    hip_pike_duration_s: float


@dataclass(frozen=True)
class Fault:
    rep_index: int
    rule: str
    message: str
    value: float
    frames: tuple[int, int]


@dataclass
class SessionResult:
    video: VideoInfo
    reps: list[Rep]
    faults: list[Fault]
