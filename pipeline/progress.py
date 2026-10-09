"""Progress across sessions: a few numbers per session, from its judged result.

Like the verdicts, these are computed when asked for, from stored measurements and the current
rules, and never stored, so a threshold change moves the whole history consistently.
"""

import math
from dataclasses import dataclass

import numpy as np

from pipeline.models import SessionResult


@dataclass(frozen=True)
class SessionProgress:
    """One session's numbers. The per-metric fields are keyed by Rep field name."""

    reps: int
    to_fix: int  # reps with at least one fault shown to the user
    clean_share: float  # share of reps with no fault shown; NaN with no reps
    average: dict[str, float]  # mean over the reps where the metric was measured
    middle_half: dict[str, tuple[float, float]]  # 25th and 75th percentile across reps
    fatigue: dict[str, float]  # last third of the set minus the first third


def session_progress(result: SessionResult, metrics: list[str]) -> SessionProgress:
    """The session's numbers, for the given Rep metrics (the web app passes the metrics of
    rules with a chart: caption)."""
    n = len(result.reps)
    to_fix = len({f.rep_index for f in result.faults if not f.suppressed})
    average, middle_half, fatigue = {}, {}, {}
    for metric in metrics:
        values = np.array([getattr(rep, metric) for rep in result.reps], dtype=float)
        measured = values[~np.isnan(values)]  # an unmeasured rep says nothing, not zero
        if measured.size:
            average[metric] = float(measured.mean())
            low, high = np.percentile(measured, [25, 75])
            middle_half[metric] = (float(low), float(high))
        else:
            average[metric] = math.nan
            middle_half[metric] = (math.nan, math.nan)
        fatigue[metric] = fatigue_index(values)
    clean_share = (n - to_fix) / n if n else math.nan
    return SessionProgress(n, to_fix, clean_share, average, middle_half, fatigue)


def fatigue_index(values: np.ndarray) -> float:
    """Mean of the last third of the reps minus the mean of the first third, in rep order.

    For depth (degrees above parallel), positive means the reps got shallower as the set went
    on. Needs at least three reps, one per third; unmeasured reps are skipped within a third,
    and a third with none measured gives NaN.
    """
    third = len(values) // 3
    if third == 0:
        return math.nan
    first, last = values[:third], values[-third:]
    if np.isnan(first).all() or np.isnan(last).all():
        return math.nan
    return float(np.nanmean(last) - np.nanmean(first))
