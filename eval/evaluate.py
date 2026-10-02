"""Phase 5: compare the pipeline's faults with the hand labels, per fault type.

    python -m eval.evaluate              # the training clips: tune against these
    python -m eval.evaluate --held-out   # the held-out clips: run once, when tuning is done
    python -m eval.evaluate --by-clip    # also break each fault down per clip

Each (rep, rule) pair has one of three outcomes: fault, ok, or could not evaluate (a metric
was NaN because a landmark was outside the frame). Could-not-evaluate is counted on its own
and left out of precision and recall: charging the detector a miss when it had no data
would measure the camera setup, not the detector.

Two tables are printed. "Detector" scores every fault the rules found. "Feedback" scores
what the user is told, after suppression (hip_pike hides shallow on the same rep).

A clip whose rep count differs from its labels is left out of the per-rep scoring and
listed as a counting failure: with a rep missing or extra, rep N of the pipeline is not rep
N of the labels, and every comparison after the mistake would be against the wrong rep.
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from pipeline.models import SessionResult
from pipeline.reps import SingleRepError
from pipeline.rules import load_rules
from pipeline.session import analyze

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels.csv"
VIDEOS = ROOT / "data" / "videos"
CONFIG = ROOT / "configs" / "pushup.yaml"
# Chosen before any detection code existed (README, "Held-out evaluation set").
HELD_OUT = {"clip10", "clip12", "clip15", "clip17", "clip18", "clip20"}


def outcomes(result: SessionResult, rules: list[str]) -> pd.DataFrame:
    """One row per (rep, rule): its outcome for the detector and for the feedback."""
    faults = {(f.rep_index, f.rule): f for f in result.faults}
    unevaluated = {(u.rep_index, u.rule) for u in result.unevaluated}
    rows = []
    for rep in result.reps:
        for rule in rules:
            key = (rep.index, rule)
            if key in unevaluated:
                detector = feedback = "unevaluated"
            elif key in faults:
                detector = "fault"
                feedback = "ok" if faults[key].suppressed else "fault"
            else:
                detector = feedback = "ok"
            rows.append(
                {"rep": rep.index, "rule": rule, "detector": detector, "feedback": feedback}
            )
    return pd.DataFrame(rows)


def score(joined: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per rule: hits, misses and false alarms against the labels, plus the reps that could
    not be evaluated (and how many of those were labelled as faults)."""
    rows = []
    for rule, group in joined.groupby("rule", sort=False):
        unknown = group[column] == "unevaluated"
        judged = group[~unknown]
        flagged = judged[column] == "fault"
        truth = judged["label"] == 1
        hits, missed = int((flagged & truth).sum()), int((~flagged & truth).sum())
        false_alarms = int((flagged & ~truth).sum())
        rows.append(
            {
                "fault": rule,
                "labelled": hits + missed,
                "detected": hits,
                "missed": missed,
                "false alarms": false_alarms,
                "precision": _ratio(hits, hits + false_alarms),
                "recall": _ratio(hits, hits + missed),
                "not evaluated": int(unknown.sum()),
                "of which labelled": int((group[unknown]["label"] == 1).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("fault")


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def by_clip(joined: pd.DataFrame, column: str) -> pd.DataFrame:
    """hits/misses/false alarms per clip and rule, to spot a threshold that only works on
    one clip."""
    judged = joined[joined[column] != "unevaluated"]
    flagged, truth = judged[column] == "fault", judged["label"] == 1
    counts = pd.DataFrame(
        {
            "clip": judged["clip"],
            "rule": judged["rule"],
            "hit": flagged & truth,
            "miss": ~flagged & truth,
            "false": flagged & ~truth,
        }
    )
    summed = counts.groupby(["clip", "rule"], sort=False).sum()
    cells = summed.apply(lambda r: f"{r['hit']}/{r['miss']}/{r['false']}", axis=1)
    return cells.unstack("rule").fillna("-")  # "-": no rep of that clip could be evaluated


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--held-out", action="store_true", help="evaluate the held-out clips")
    parser.add_argument("--by-clip", action="store_true", help="also show each clip")
    args = parser.parse_args(argv)

    rules = [rule.name for rule in load_rules(CONFIG)]
    labels = pd.read_csv(LABELS)
    missing = set(rules) - set(labels.columns)
    if missing:
        print(f"rules without a labels.csv column: {sorted(missing)}", file=sys.stderr)
        return 1

    clips = [c for c in sorted(labels["clip"].unique()) if (c in HELD_OUT) == args.held_out]
    predictions, failures = [], []
    for clip in clips:
        labelled = int((labels["clip"] == clip).sum())
        try:
            result = analyze(VIDEOS / f"{clip}.mp4", CONFIG)
        except SingleRepError:
            failures.append(f"{clip}: 1 rep found, {labelled} labelled")
            continue
        if len(result.reps) != labelled:
            failures.append(f"{clip}: {len(result.reps)} reps found, {labelled} labelled")
            continue
        predictions.append(outcomes(result, rules).assign(clip=clip))
    if not predictions:  # pd.concat of nothing raises; say why instead
        print("No clip could be scored: every rep count was wrong.", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    truth = labels[labels["clip"].isin(clips)].melt(
        id_vars=["clip", "rep"], value_vars=rules, var_name="rule", value_name="label"
    )
    joined = pd.concat(predictions).merge(truth, on=["clip", "rep", "rule"], how="left")

    scored_clips = joined["clip"].nunique()
    scored_reps = len(joined[["clip", "rep"]].drop_duplicates())
    print(f"{'Held-out' if args.held_out else 'Training'} clips: {len(clips)}")
    print(f"Rep counts correct: {len(clips) - len(failures)} of {len(clips)} clips")
    for failure in failures:
        print(f"  left out, {failure}")
    print(f"Scored: {scored_reps} reps from {scored_clips} clips")
    print("Reps within a clip share a camera and a body: they are not independent samples.\n")
    for column, title in (("detector", "Detector"), ("feedback", "Feedback (after suppression)")):
        print(title)
        print(score(joined, column).to_string(float_format=lambda x: f"{x:.2f}", na_rep="-"))
        print()
    if args.by_clip:
        print("Per clip, hits/misses/false alarms (detector, could-not-evaluate left out)")
        print(by_clip(joined, "detector").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
