"""Core dataclasses passed between pipeline stages: VideoInfo, Rep, Fault, Unevaluated,
SessionResult.

A type lives here when one module creates it and another uses it: these are the contracts
between stages. A type used only inside one module (its settings, like RepConfig and Rule,
or internal bundles, like ClipSignals) lives next to its code. This file imports only
dataclasses, so every module can import it and it can never be part of an import cycle.
"""

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

    Every metric a rule can use has a `<metric>_frame`: the frame its value came from, so a
    fault can be shown at the moment it happened. rules.load_rules checks the pairing.
    """

    index: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    min_elbow_angle: float  # depth, elbow-based (diagnostic; shallow uses the upper arm)
    min_upper_arm_angle: float  # 0 = upper arm parallel to floor (shallow)
    max_elbow_angle: float  # lockout at the top (no_lockout)
    max_hip_drop: float  # hip below the body line (hip_sag)
    max_hip_rise: float  # hip above the body line (hip_pike)
    hip_sag_duration_s: float
    hip_pike_duration_s: float
    min_elbow_angle_frame: int
    min_upper_arm_angle_frame: int  # always bottom_frame; kept so every metric has a frame
    max_elbow_angle_frame: int  # the top the rep rises to
    max_hip_drop_frame: int
    max_hip_rise_frame: int


@dataclass(frozen=True)
class Fault:
    """A rule the rep broke. frames: the whole rep; frame: the moment the measured value
    came from (the bottom for depth, the top for lockout). suppressed: a higher-priority
    fault on the same rep hides it from the feedback; it is kept so evaluation can still
    score the raw detector."""

    rep_index: int
    rule: str
    message: str
    value: float
    frames: tuple[int, int]
    frame: int
    suppressed: bool = False


@dataclass(frozen=True)
class Unevaluated:
    """A rule that could not be checked on a rep, because its metric is NaN (a landmark it
    needs was outside the frame). Not a fault, and not a pass either."""

    rep_index: int
    rule: str


@dataclass
class SessionResult:
    video: VideoInfo
    reps: list[Rep]
    faults: list[Fault]
    unevaluated: list[Unevaluated]
