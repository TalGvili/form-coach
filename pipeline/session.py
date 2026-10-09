"""From a video file to a SessionResult: the pipeline stages in order, and nothing else.

Both entry points call analyze(): analyze.py on the command line, and the web app later.
It returns data and lets errors propagate (a missing file, a pipeline.errors.AnalysisError);
printing, saving, and turning errors into messages or HTTP responses is each caller's job.
"""

import dataclasses
import math
from pathlib import Path
from typing import Any

from pipeline.landmarks import load_or_extract
from pipeline.models import Rep, SessionResult, VideoInfo
from pipeline.reps import load_config, segment_reps
from pipeline.rules import Rule, evaluate, load_rules, mark_suppressed


def analyze(video: str | Path, config: Path) -> SessionResult:
    """Landmarks (cached after the first run) -> reps -> rule checks -> feedback priority."""
    info, landmarks = load_or_extract(video)
    reps = segment_reps(landmarks, info, load_config(config))
    return judge(info, reps, load_rules(config))


def judge(video: VideoInfo, reps: list[Rep], rules: list[Rule]) -> SessionResult:
    """The verdicts on measured reps: rule checks, then feedback priority.

    The one place verdicts are made. A new analysis calls it, and so does reading a stored
    session, which keeps only measurements: an old session is judged by today's rules."""
    faults, unevaluated = evaluate(reps, rules)
    return SessionResult(video, reps, mark_suppressed(faults, rules), unevaluated)


def to_json(result: Any) -> dict[str, Any]:
    """A result dataclass (a SessionResult, a SessionProgress) as plain dicts and lists, ready
    for json.dump.

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
