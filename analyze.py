"""Command line: analyze one video, print a per-rep report and save it as JSON.

    python analyze.py data/videos/clip04.mp4 [--render]

All the work happens in pipeline.session.analyze; this file only reads the arguments and
presents the result. --render also writes an annotated video (pipeline.render).
"""

import argparse
import json
import sys
from pathlib import Path

from pipeline.models import SessionResult
from pipeline.render import render
from pipeline.reps import SingleRepError
from pipeline.rules import load_rules
from pipeline.session import analyze, to_json

REPORTS_DIR = Path(__file__).resolve().parent / "data" / "reports"


def print_report(result: SessionResult) -> None:
    """One line per rep: its time span, the rules it broke, and those it couldn't check.
    A suppressed fault is listed apart: it isn't feedback, but the report hides nothing."""
    fps = result.video.fps
    shown = [f for f in result.faults if not f.suppressed]
    print(
        f"{result.video.path}: {len(result.reps)} reps, {len(shown)} faults,"
        f" {len(result.unevaluated)} checks could not be evaluated"
    )
    for rep in result.reps:
        broken = [f.rule for f in shown if f.rep_index == rep.index]
        hidden = [f.rule for f in result.faults if f.rep_index == rep.index and f.suppressed]
        unknown = [u.rule for u in result.unevaluated if u.rep_index == rep.index]
        span = f"{rep.start_frame / fps:.1f}-{rep.end_frame / fps:.1f} s"
        line = f"  rep {rep.index:>2}  {span:<13} {', '.join(broken) or 'clean'}"
        if hidden:
            line += f"  (suppressed: {', '.join(hidden)})"
        if unknown:
            line += f"  (could not evaluate: {', '.join(unknown)})"
        print(line)
    messages = dict.fromkeys(f.message for f in shown)  # unique, in order
    for message in messages:
        print(f"  - {message}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/pushup.yaml"))
    parser.add_argument(
        "--render", action="store_true", help="also write an annotated video (slower)"
    )
    args = parser.parse_args(argv)
    if not args.video.is_file():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 1

    try:
        result = analyze(args.video, args.config)
    except SingleRepError as error:
        print(f"{args.video}: {error}", file=sys.stderr)
        return 1

    print_report(result)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / f"{args.video.stem}.json"
    report.write_text(json.dumps(to_json(result), indent=2), encoding="utf-8")
    print(f"saved {report}")
    if args.render:
        annotated = REPORTS_DIR / f"{args.video.stem}_annotated.mp4"
        titles = {rule.name: rule.title for rule in load_rules(args.config)}
        render(result, titles, annotated)
        print(f"saved {annotated}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
