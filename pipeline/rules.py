"""Rule engine: check every rep against the rules in a config file.

The engine knows nothing about push-ups. A rule names a Rep field and the bounds it must
stay inside; everything exercise-specific lives in the YAML. A new exercise is a new
config file, not new code here.

Each check has three outcomes: a fault, no fault, or could not evaluate. The third is not
optional: a metric is NaN when a landmark it needs was outside the frame, and `NaN > 15` is
False, so without it an unmeasured rep would silently count as clean.
"""

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

from pipeline.models import Fault, Rep, Unevaluated

REP_FIELDS = {f.name for f in fields(Rep)}


@dataclass(frozen=True)
class Rule:
    """One entry of the config's `rules:` section. The name is the YAML key and matches the
    labels.csv column; the title is how the check is named to a user ("Arm lockout")."""

    name: str
    metric: str
    message: str
    title: str
    min: float | None = None
    max: float | None = None
    duration_metric: str | None = None
    min_duration_s: float | None = None
    suppressed_by: tuple[str, ...] = ()  # rules whose fault on the same rep hides this one
    chart: str | None = None  # a per-rep chart of the metric on the web page, with this caption


def load_rules(path: Path) -> list[Rule]:
    """Read the `rules:` section, failing at load time rather than halfway through a video.

    An unknown key raises a TypeError from the Rule constructor. The checks below catch the
    mistakes the constructor can't: a metric that is not a Rep field or has no `_frame`
    partner, a rule with no bounds, a duration half-specified, and suppressed_by naming a
    rule that doesn't exist.
    """
    with path.open(encoding="utf-8") as handle:
        section = yaml.safe_load(handle)["rules"]
    rules = [
        Rule(name=name, **{**spec, "suppressed_by": tuple(spec.get("suppressed_by", ()))})
        for name, spec in section.items()
    ]
    names = {rule.name for rule in rules}
    for rule in rules:
        for other in rule.suppressed_by:
            if other not in names or other == rule.name:
                raise ValueError(f"rule {rule.name!r}: can't be suppressed by {other!r}")
        for metric in (rule.metric, rule.duration_metric):
            if metric is not None and metric not in REP_FIELDS:
                raise ValueError(f"rule {rule.name!r}: {metric!r} is not a Rep field")
        if f"{rule.metric}_frame" not in REP_FIELDS:
            raise ValueError(f"rule {rule.name!r}: Rep has no {rule.metric}_frame")
        if rule.min is None and rule.max is None:
            raise ValueError(f"rule {rule.name!r}: needs min, max or both")
        if (rule.duration_metric is None) != (rule.min_duration_s is None):
            raise ValueError(f"rule {rule.name!r}: duration_metric and min_duration_s go together")
        if rule.chart and rule.duration_metric:
            # a bar past the line that didn't last long enough would look like a missed fault
            raise ValueError(f"rule {rule.name!r}: a chart can't show a duration condition")
    return rules


def check(rule: Rule, rep: Rep) -> Fault | Unevaluated | None:
    """Apply one rule to one rep. None means the rep passed."""
    value = getattr(rep, rule.metric)
    if math.isnan(value):
        return Unevaluated(rep.index, rule.name)
    too_high = rule.max is not None and value > rule.max
    too_low = rule.min is not None and value < rule.min
    if not (too_high or too_low):
        return None
    if rule.duration_metric is not None:
        duration = getattr(rep, rule.duration_metric)
        if math.isnan(duration):
            return Unevaluated(rep.index, rule.name)
        if duration < rule.min_duration_s:
            return None  # past the bound, but too briefly to count
    return Fault(
        rep_index=rep.index,
        rule=rule.name,
        message=rule.message,
        value=value,
        frames=(rep.start_frame, rep.end_frame),
        frame=getattr(rep, f"{rule.metric}_frame"),  # when it happened
    )


def evaluate(reps: list[Rep], rules: list[Rule]) -> tuple[list[Fault], list[Unevaluated]]:
    """Every rule on every rep. A (rep, rule) pair in neither list passed."""
    faults: list[Fault] = []
    unevaluated: list[Unevaluated] = []
    for rep in reps:
        for rule in rules:
            result = check(rule, rep)
            if isinstance(result, Fault):
                faults.append(result)
            elif isinstance(result, Unevaluated):
                unevaluated.append(result)
    return faults, unevaluated


def mark_suppressed(faults: list[Fault], rules: list[Rule]) -> list[Fault]:
    """Mark each fault that a higher-priority fault on the same rep hides from the feedback.

    Coaches correct one thing at a time. Nothing is removed: evaluate() already judged every
    rule, and evaluation scores both the raw detector and the feedback. Only a fault
    suppresses; an unevaluated rule is not evidence of anything.
    """
    fired = {(fault.rep_index, fault.rule) for fault in faults}
    hidden_by = {rule.name: rule.suppressed_by for rule in rules}
    return [
        replace(fault, suppressed=True)
        if any((fault.rep_index, other) in fired for other in hidden_by[fault.rule])
        else fault
        for fault in faults
    ]
