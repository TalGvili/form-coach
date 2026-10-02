"""Tests for eval.evaluate's scoring, on a hand-made result where every count is known."""

import dataclasses
import math

import pandas as pd

from eval.evaluate import outcomes, score
from pipeline.models import Fault, Rep, SessionResult, Unevaluated, VideoInfo

REP = Rep(1, 0, 30, 60, *[0.0] * 7, *[0] * 5)
VIDEO = VideoInfo(path="clip.mp4", fps=30.0, width=1920, height=1080, n_frames=200)
RULES = ["shallow", "hip_pike", "no_lockout"]


def fault(rep: int, rule: str, suppressed: bool = False) -> Fault:
    return Fault(rep, rule, "m", 0.0, (0, 60), frame=30, suppressed=suppressed)


# rep 1: shallow (hidden by a pike) and pike; rep 2: lockout not measurable; rep 3: clean
RESULT = SessionResult(
    VIDEO,
    [dataclasses.replace(REP, index=i) for i in (1, 2, 3)],
    [fault(1, "shallow", suppressed=True), fault(1, "hip_pike")],
    [Unevaluated(2, "no_lockout")],
)


def outcome(table: pd.DataFrame, rep: int, rule: str, column: str) -> str:
    return table[(table["rep"] == rep) & (table["rule"] == rule)][column].item()


class TestOutcomes:
    def test_every_rep_gets_every_rule(self):
        assert len(outcomes(RESULT, RULES)) == 3 * len(RULES)

    def test_a_suppressed_fault_is_a_fault_for_the_detector_but_not_the_feedback(self):
        table = outcomes(RESULT, RULES)
        assert outcome(table, 1, "shallow", "detector") == "fault"
        assert outcome(table, 1, "shallow", "feedback") == "ok"

    def test_unevaluated_stays_unevaluated_in_both(self):
        table = outcomes(RESULT, RULES)
        assert outcome(table, 2, "no_lockout", "detector") == "unevaluated"
        assert outcome(table, 2, "no_lockout", "feedback") == "unevaluated"


def joined(labels: dict[tuple[int, str], int]) -> pd.DataFrame:
    table = outcomes(RESULT, RULES).assign(clip="clip01")
    pairs = zip(table["rep"], table["rule"], strict=True)
    table["label"] = [labels.get(pair, 0) for pair in pairs]
    return table


def test_counts_and_ratios():
    # labels: rep 1 really is shallow; rep 3 really is piked; rep 2 really has no lockout
    table = joined({(1, "shallow"): 1, (3, "hip_pike"): 1, (2, "no_lockout"): 1})
    detector = score(table, "detector")
    assert detector.loc["shallow", ["detected", "missed", "false alarms"]].tolist() == [1, 0, 0]
    # pike: found on rep 1 (labelled clean), missed on rep 3
    assert detector.loc["hip_pike", ["detected", "missed", "false alarms"]].tolist() == [0, 1, 1]
    assert detector.loc["hip_pike", "precision"] == 0.0


def test_suppression_costs_a_hit_in_the_feedback_table():
    table = joined({(1, "shallow"): 1})
    assert score(table, "detector").loc["shallow", "recall"] == 1.0
    assert score(table, "feedback").loc["shallow", "recall"] == 0.0


def test_could_not_evaluate_is_counted_apart_not_as_a_miss():
    no_lockout = score(joined({(2, "no_lockout"): 1}), "detector").loc["no_lockout"]
    assert no_lockout["missed"] == 0
    assert no_lockout["not evaluated"] == 1
    assert no_lockout["of which labelled"] == 1
    assert math.isnan(no_lockout["recall"])  # nothing labelled was judged: no recall to report
