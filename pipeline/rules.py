"""Rule engine: check every rep against the rules in a config file.

The engine knows nothing about push-ups. A rule names a Rep field and the bounds it must
stay inside; everything exercise-specific lives in the YAML. A new exercise is a new
config file, not new code here.

Each check has three outcomes: a fault, no fault, or could not evaluate. The third is not
optional: a metric is NaN when a landmark it needs was outside the frame, and `NaN > 15` is
False, so without it an unmeasured rep would silently count as clean.
"""

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from pipeline.models import Fault, Rep, Unevaluated

REP_FIELDS = {f.name for f in fields(Rep)}


@dataclass(frozen=True)
class Rule:
    """One entry of the config's `rules:` section. The name is the YAML key."""

    name: str
    metric: str
    message: str
    min: float | None = None
    max: float | None = None
    duration_metric: str | None = None
    min_duration_s: float | None = None


def load_rules(path: Path) -> list[Rule]:
    """Read the `rules:` section, failing at load time rather than halfway through a video.

    An unknown key raises a TypeError from the Rule constructor. The checks below catch the
    mistakes the constructor can't: a metric that is not a Rep field, a rule with no bounds,
    and a duration half-specified.
    """
    with path.open(encoding="utf-8") as handle:
        section = yaml.safe_load(handle)["rules"]
    rules = [Rule(name=name, **spec) for name, spec in section.items()]
    for rule in rules:
        for metric in (rule.metric, rule.duration_metric):
            if metric is not None and metric not in REP_FIELDS:
                raise ValueError(f"rule {rule.name!r}: {metric!r} is not a Rep field")
        if rule.min is None and rule.max is None:
            raise ValueError(f"rule {rule.name!r}: needs min, max or both")
        if (rule.duration_metric is None) != (rule.min_duration_s is None):
            raise ValueError(f"rule {rule.name!r}: duration_metric and min_duration_s go together")
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
    return Fault(rep.index, rule.name, rule.message, value, (rep.start_frame, rep.end_frame))


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
