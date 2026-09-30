"""From a video file to a SessionResult: the pipeline stages in order, and nothing else.

Both entry points call analyze(): analyze.py on the command line, and the web app later.
It returns data and lets errors propagate (a missing file, SingleRepError); printing,
saving, and turning errors into messages or HTTP responses is each caller's job.
"""

import dataclasses
import math
from pathlib import Path
from typing import Any

from pipeline.landmarks import load_or_extract
from pipeline.models import SessionResult
from pipeline.reps import load_config, segment_reps
from pipeline.rules import evaluate, load_rules, mark_suppressed


def analyze(video: str | Path, config: Path) -> SessionResult:
    """Landmarks (cached after the first run) -> reps -> rule checks -> feedback priority."""
    info, landmarks = load_or_extract(video)
    reps = segment_reps(landmarks, info, load_config(config))
    rules = load_rules(config)
    faults, unevaluated = evaluate(reps, rules)
    return SessionResult(info, reps, mark_suppressed(faults, rules), unevaluated)


def to_json(result: SessionResult) -> dict[str, Any]:
    """The result as plain dicts and lists, ready for json.dump.

    NaN ("could not measure") becomes None. json.dump would write NaN, which is not valid
    JSON: a browser's JSON.parse rejects the whole document.
    """
    return _nan_to_none(dataclasses.asdict(result))


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _nan_to_none(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_nan_to_none(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
