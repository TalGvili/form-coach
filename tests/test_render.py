"""Tests for what the annotated video pauses on and summarises. Drawing itself needs a video
file; deciding what to show is pure logic and is tested here."""

import dataclasses

from pipeline.models import Fault, Rep, SessionResult, Unevaluated, VideoInfo
from pipeline.render import _pauses, _unchecked_summary

SHALLOW = Fault(1, "shallow", "Not deep enough", 20.0, (0, 60), frame=30)  # at the bottom
SAG = Fault(1, "hip_sag", "Hips dropping", 25.0, (0, 60), frame=30)  # also at the bottom
LOCKOUT = Fault(1, "no_lockout", "Straighten your arms", 140.0, (0, 60), frame=58)  # the top


class TestPauses:
    def test_a_single_fault_pauses_on_the_frame_it_came_from(self):
        assert _pauses([LOCKOUT]) == {58: [LOCKOUT]}

    def test_a_reps_faults_share_one_pause_at_the_last_of_them(self):
        """clip04: a shallow bottom then a sag on the way up froze the video twice per rep.
        One pause, at the later moment, so neither message comes before its fault."""
        later_sag = dataclasses.replace(SAG, frame=45)
        assert _pauses([later_sag, SHALLOW]) == {45: [SHALLOW, later_sag]}

    def test_faults_of_different_reps_pause_separately(self):
        next_rep = dataclasses.replace(SHALLOW, rep_index=2, frame=90)
        assert _pauses([SHALLOW, next_rep]) == {30: [SHALLOW], 90: [next_rep]}

    def test_a_suppressed_fault_does_not_pause(self):
        hidden = dataclasses.replace(SHALLOW, suppressed=True)
        assert _pauses([hidden, LOCKOUT]) == {58: [LOCKOUT]}


def test_unchecked_rules_are_summarised_once_per_rule_in_the_users_words():
    rep = Rep(1, 0, 30, 60, *[0.0] * 7, *[0] * 5)
    reps = [dataclasses.replace(rep, index=i) for i in (1, 2, 3)]
    video = VideoInfo(path="clip.mp4", fps=30.0, width=1920, height=1080, n_frames=200)
    unevaluated = [Unevaluated(i, "no_lockout") for i in (1, 2, 3)] + [Unevaluated(3, "shallow")]
    result = SessionResult(video, reps, [], unevaluated)
    titles = {"no_lockout": "Arm lockout", "shallow": "Depth"}
    assert _unchecked_summary(result, titles) == ["Arm lockout: 3 of 3 reps", "Depth: 1 of 3 reps"]
