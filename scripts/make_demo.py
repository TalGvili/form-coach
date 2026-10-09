"""Build the static demo's data from the private dataset. Run locally, then commit demo/.

    python scripts/make_demo.py [clip]        # clip09 by default

demo/report.json    the report the web app returns for the clip, as the page reads it
demo/video.mp4      its annotated video, re-encoded smaller for the web (540 px, crf 30)
demo/progress.json  the history page's numbers for person A's training sessions: numbers only,
                    no video, with made-up dates three days apart

clip09 is the default because it alternates deep and shallow reps (25-37 deg against 3-12, with
the limit at 15), every rep agrees with its label, and none carries a borderline note.
"""

import json
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


def write_report(clip: str) -> None:
    result = analyze(VIDEOS / f"{clip}.mp4", main.CONFIG)
    with tempfile.TemporaryDirectory() as scratch:
        full = Path(scratch) / "annotated.mp4"
        with main.h264_writer(full, result.video) as write:
            starts = annotate(result, main.TITLES, write)
        # smaller than the app's copy: a web page shouldn't load 17 MB before it plays
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(full), "-c:v", "libx264"]
        command += ["-preset", "slow", "-crf", "30", "-vf", "scale=-2:540"]
        command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        command += ["-an", str(DEMO / "video.mp4")]
        subprocess.run(command, check=True)
    report = main.report_for(result, "demo", starts, session_id=None)
    report["video_url"] = "demo/video.mp4"  # relative: the site isn't served from /
    (DEMO / "report.json").write_text(json.dumps(report), encoding="utf-8")


def write_progress(demo_clip: str) -> None:
    start = datetime(2026, 9, 1, 18, 0, tzinfo=UTC)
    sessions = []
    for day, clip in enumerate(HISTORY):
        result = analyze(VIDEOS / f"{clip}.mp4", main.CONFIG)
        progress = to_json(session_progress(result, list(main.CHARTED)))
        when = (start + timedelta(days=3 * day)).isoformat(timespec="seconds")
        entry = {"id": day + 1, "created_at": when, **progress}
        entry["demo_report"] = clip == demo_clip  # the one session whose report is bundled
        sessions.append(entry)
    progress = {"metrics": main.CHARTED, "sessions": sessions}
    (DEMO / "progress.json").write_text(json.dumps(progress), encoding="utf-8")


def main_() -> None:
    clip = sys.argv[1] if len(sys.argv) > 1 else "clip09"
    DEMO.mkdir(exist_ok=True)
    write_report(clip)
    write_progress(clip)
    for path in sorted(DEMO.iterdir()):
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main_()
