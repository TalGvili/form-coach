"""Tests for pipeline.rules, on hand-made Reps where the right outcome is obvious."""

import dataclasses
import math
from pathlib import Path

import pytest
import yaml

from pipeline.models import Fault, Rep, Unevaluated
from pipeline.reps import load_config
from pipeline.rules import Rule, check, evaluate, load_rules

CLEAN = Rep(
    index=1,
    start_frame=10,
    bottom_frame=40,
    end_frame=70,
    min_elbow_angle=85.0,
    min_upper_arm_angle=2.0,
    max_elbow_angle=175.0,
    max_hip_drop=3.0,
    max_hip_rise=0.0,
    hip_sag_duration_s=0.0,
    hip_pike_duration_s=0.0,
)
SHALLOW = Rule("shallow", "min_upper_arm_angle", "Not deep enough", max=5)
NO_LOCKOUT = Rule("no_lockout", "max_elbow_angle", "Straighten your arms", min=160)
HIP_SAG = Rule(
    "hip_sag",
    "max_hip_drop",
    "Hips dropping",
    max=15,
    duration_metric="hip_sag_duration_s",
    min_duration_s=0.3,
)


def rep(**changes: float) -> Rep:
    return dataclasses.replace(CLEAN, **changes)


class TestCheck:
    def test_a_clean_rep_passes_every_rule(self):
        assert all(check(rule, CLEAN) is None for rule in (SHALLOW, NO_LOCKOUT, HIP_SAG))

    def test_above_max_is_a_fault_on_the_rep_frames(self):
        fault = check(SHALLOW, rep(min_upper_arm_angle=20.0))
        assert fault == Fault(1, "shallow", "Not deep enough", 20.0, (10, 70))

    def test_below_min_is_a_fault(self):
        assert isinstance(check(NO_LOCKOUT, rep(max_elbow_angle=140.0)), Fault)

    def test_exactly_on_the_bound_passes(self):
        assert check(SHALLOW, rep(min_upper_arm_angle=5.0)) is None
        assert check(NO_LOCKOUT, rep(max_elbow_angle=160.0)) is None

    def test_nan_is_unevaluated_not_a_pass(self):
        """The case the whole third outcome exists for: NaN > 5 is False."""
        assert check(SHALLOW, rep(min_upper_arm_angle=math.nan)) == Unevaluated(1, "shallow")

    def test_past_the_bound_too_briefly_is_not_a_fault(self):
        assert check(HIP_SAG, rep(max_hip_drop=25.0, hip_sag_duration_s=0.1)) is None

    def test_past_the_bound_long_enough_is_a_fault(self):
        assert isinstance(check(HIP_SAG, rep(max_hip_drop=25.0, hip_sag_duration_s=0.6)), Fault)

    def test_unknown_duration_is_unevaluated(self):
        result = check(HIP_SAG, rep(max_hip_drop=25.0, hip_sag_duration_s=math.nan))
        assert result == Unevaluated(1, "hip_sag")


def test_evaluate_sorts_every_rep_and_rule_into_three_outcomes():
    reps = [
        CLEAN,
        rep(index=2, min_upper_arm_angle=30.0),  # shallow
        rep(index=3, max_elbow_angle=math.nan),  # lockout unmeasurable
    ]
    faults, unevaluated = evaluate(reps, [SHALLOW, NO_LOCKOUT])
    assert [(f.rep_index, f.rule) for f in faults] == [(2, "shallow")]
    assert unevaluated == [Unevaluated(3, "no_lockout")]


class TestLoadRules:
    def test_the_config_file_loads(self, config_path: Path):
        rules = load_rules(config_path)
        assert [r.name for r in rules] == ["shallow", "hip_sag", "hip_pike", "no_lockout"]

    def test_hip_rules_use_the_threshold_durations_are_measured_at(self, config_path: Path):
        """Durations count time past hip_duration_threshold_deg; a rule with a different max
        would compare its magnitude against one number and its duration against another."""
        hip = {r.name: r for r in load_rules(config_path)}
        threshold = load_config(config_path).hip_duration_threshold_deg
        assert hip["hip_sag"].max == hip["hip_pike"].max == threshold

    @pytest.mark.parametrize(
        ("spec", "complaint"),
        [
            ({"metric": "max_hip_dorp", "max": 15, "message": "m"}, "not a Rep field"),
            ({"metric": "max_hip_drop", "message": "m"}, "needs min, max"),
            (
                {"metric": "max_hip_drop", "max": 15, "min_duration_s": 0.3, "message": "m"},
                "go together",
            ),
        ],
    )
    def test_a_broken_rule_fails_at_load_time(self, tmp_path: Path, spec: dict, complaint: str):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"rules": {"broken": spec}}), encoding="utf-8")
        with pytest.raises(ValueError, match=complaint):
            load_rules(path)
