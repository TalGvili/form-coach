"""Build the static demo's data from the private dataset. Run locally, then commit demo/.

    python scripts/make_demo.py

demo/examples.json  the examples the page offers: key, title and a one-line caption
demo/<key>.json     an example's report, as the web app returns it
demo/<key>.mp4      its annotated video, re-encoded smaller for the web (540 px, crf 30)
demo/progress.json  the history page's numbers for person A's training sessions: numbers only,
                    no video, with made-up dates three days apart

The clips were filmed one fault at a time, so no single clip shows every fault cleanly. Each
example is a clip where every rep agrees with its label, none carries a borderline note, and
every shown value is far from its limit:
  clip09  deep and shallow reps alternate: depth 25-37 deg against 3-12, limit 15
  clip16  locked and bent arms alternate: 122-151 deg against 164-172, limit 156
  clip19  person B, filmed facing the other way: hips sag 21-30 deg (limit 13) for over a
          second, and depth 35-47 deg, on every rep
Hip pike has no example: clip06 has three relabelled borderline reps, and on clip11 a rep's
depth reads above its limit but isn't flagged (the pike hides it), which the chart would show
without explaining. Shown with permission from both people filmed.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import main  # noqa: E402  (after the path fix, so the script runs from anywhere)
from pipeline.progress import session_progress  # noqa: E402
from pipeline.render import annotate  # noqa: E402
from pipeline.session import analyze, to_json  # noqa: E402

VIDEOS = ROOT / "data" / "videos"
DEMO = ROOT / "demo"
# Person A's training clips with a correct rep count: one person's history. Held-out clips and
# person B (clip18-20) are left out.
HISTORY = ["clip01", "clip02", "clip03", "clip04", "clip05", "clip06", "clip07", "clip09",
           "clip11", "clip14", "clip16"]  # fmt: skip
# key (file names and ?example= on the page), clip, title, caption
EXAMPLES = [
    ("depth", "clip09", "Depth", "Every other rep stops short of parallel."),
    ("lockout", "clip16", "Arm lockout", "Every other rep stops before the arms are straight."),
    (
        "hips",
        "clip19",
        "Hip sag",
        "Another person, filmed facing the other way: the hips sag and the reps are shallow "
        "on every rep.",
    ),
]


def write_report(key: str, clip: str) -> None:
    result = analyze(VIDEOS / f"{clip}.mp4", main.CONFIG)
    with tempfile.TemporaryDirectory() as scratch:
        full = Path(scratch) / "annotated.mp4"
        with main.h264_writer(full, result.video) as write:
            starts = annotate(result, main.TITLES, write)
        # smaller than the app's copy: a web page shouldn't load 17 MB before it plays
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(full), "-c:v", "libx264"]
        command += ["-preset", "slow", "-crf", "30", "-vf", "scale=-2:540"]
        command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        command += ["-an", str(DEMO / f"{key}.mp4")]
        subprocess.run(command, check=True)
    report = main.report_for(result, "demo", starts, session_id=None)
    report["video_url"] = f"demo/{key}.mp4"  # relative: the site isn't served from /
    (DEMO / f"{key}.json").write_text(json.dumps(report), encoding="utf-8")


def write_progress() -> None:
    example_of = {clip: key for key, clip, _, _ in EXAMPLES}
    start = datetime(2026, 9, 1, 18, 0, tzinfo=UTC)
    sessions = []
    for day, clip in enumerate(HISTORY):
        result = analyze(VIDEOS / f"{clip}.mp4", main.CONFIG)
        progress = to_json(session_progress(result, list(main.CHARTED)))
        when = (start + timedelta(days=3 * day)).isoformat(timespec="seconds")
        entry = {"id": day + 1, "created_at": when, **progress}
        entry["demo_report"] = example_of.get(clip)  # the example to open, if it is one
        sessions.append(entry)
    progress = {"metrics": main.CHARTED, "sessions": sessions}
    (DEMO / "progress.json").write_text(json.dumps(progress), encoding="utf-8")


def main_() -> None:
    shutil.rmtree(DEMO, ignore_errors=True)  # no file of an example that was dropped stays
    DEMO.mkdir()
    for key, clip, _, _ in EXAMPLES:
        write_report(key, clip)
    examples = [{"key": k, "title": t, "caption": c} for k, _, t, c in EXAMPLES]
    (DEMO / "examples.json").write_text(json.dumps(examples), encoding="utf-8")
    write_progress()
    for path in sorted(DEMO.iterdir()):
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main_()
