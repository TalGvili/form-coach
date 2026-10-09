"""Annotated video: what the analysis saw, drawn on every frame.

It draws only the five landmarks the pipeline measures (shoulder, elbow, wrist, hip, ankle
on the side facing the camera) and the shoulder-ankle line the hip is measured against, so
the viewer sees exactly what the numbers came from.

During playback the panel shows only the rep counter. Each fault has a moment: the frame
its value came from (the bottom for depth, the top for lockout, the worst point for the
hips). When the video reaches a rep's last fault moment, it pauses there for PAUSE_S
seconds with that rep's messages and a red border, then plays on. At most one pause per
rep, on a picture of the fault, and never before a fault happened.

Checks that could not be evaluated are a fact about the camera setup (a wrist out of
frame), not an event in a rep. The video opens frozen on a card that lists them, in amber,
centred, for INTRO_S seconds: a note in a corner is easy to miss, and a user who misses it
reads "no lockout fault" as "good lockout".

The pauses make the annotated video longer than the original: about PAUSE_S per faulty rep
plus the opening card. Anything synced to video time later (a chart, "jump to rep") must use
the original's.

annotate() draws the frames and hands each one to a write function, so the caller decides
where they go. render() writes them to a file with OpenCV's mp4v codec, which video players
handle but browsers don't; the web app pipes them into ffmpeg as H.264 instead, encoding
each frame once.
"""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from pipeline import signals
from pipeline.landmarks import load_or_extract
from pipeline.models import Fault, Rep, SessionResult

# OpenCV colours are BGR, not RGB.
NEUTRAL = (230, 230, 230)
FAULT = (60, 60, 230)
WARNING = (0, 190, 255)  # amber: not a fault, but something the user must not miss
PANEL = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX  # the built-in fonts draw ASCII only
PAUSE_S = 1.0  # how long the video freezes on a fault; long enough to read one line
INTRO_S = 3.5  # how long the opening card stays; long enough to read four lines
CARD_HEADING = "Not checked in this video"
CARD_ADVICE = [
    "The camera couldn't see every body part it needs.",
    "Keep your whole body in the frame.",
]


def render(result: SessionResult, titles: dict[str, str], out: Path) -> None:
    """Write the annotated video to an mp4v file: for the command line, which needs no
    ffmpeg. titles: rule name -> what a user calls the check ("no_lockout" -> "Arm lockout")."""
    video = result.video
    size = (video.width, video.height)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), video.fps, size)
    try:
        annotate(result, titles, writer.write)
    finally:
        writer.release()


def annotate(
    result: SessionResult, titles: dict[str, str], write: Callable[[np.ndarray], object]
) -> dict[int, float]:
    """Draw every frame of the annotated video, in order, and pass each to write: one call
    per frame of the video, so a pause is the same frame written PAUSE_S * fps times.

    Returns where each rep starts in the annotated video, in seconds, by rep index. The
    opening card and the pauses push every rep later than in the original, so a page that
    jumps to a rep needs these times. They are counted from the frames actually written,
    so they can't drift from the video."""
    info, landmarks = load_or_extract(result.video.path)  # a cache hit after analyze()
    side = signals.pick_side(landmarks)
    capture = cv2.VideoCapture(info.path)
    pauses = _pauses(result.faults)
    unchecked = _unchecked_summary(result, titles)
    rep_starting_at = {rep.start_frame: rep.index for rep in result.reps}
    starts: dict[int, float] = {}
    written = 0

    def emit(frame: np.ndarray) -> None:
        nonlocal written  # the counter lives in annotate, not in this inner function
        write(frame)
        written += 1

    try:
        for frame_index in range(len(landmarks)):
            ok, frame = capture.read()
            if not ok:
                break
            points = landmarks[frame_index]

            if frame_index == 0 and unchecked:
                card = frame.copy()
                lines = [(CARD_HEADING, WARNING)]
                lines += [(text, NEUTRAL) for text in unchecked + CARD_ADVICE]
                _draw_panel(card, lines, size=1.4, middle=True)
                _draw_border(card, WARNING)
                for _ in range(round(INTRO_S * info.fps)):
                    emit(card)

            current = _rep_at(result.reps, frame_index)
            playing = frame.copy()
            _draw_skeleton(playing, points, side, NEUTRAL)
            _draw_panel(playing, [(f"Rep {current.index}", NEUTRAL)] if current else [])
            if frame_index in rep_starting_at:
                starts[rep_starting_at[frame_index]] = written / info.fps
            emit(playing)

            if frame_index in pauses:
                paused = frame.copy()
                _draw_skeleton(paused, points, side, FAULT)
                messages = [(f"Rep {f.rep_index}: {f.message}", FAULT) for f in pauses[frame_index]]
                _draw_panel(paused, messages)
                _draw_border(paused, FAULT)
                for _ in range(round(PAUSE_S * info.fps)):
                    emit(paused)
    finally:
        capture.release()
    return starts


def _rep_at(reps: list[Rep], frame_index: int) -> Rep | None:
    """The rep in progress. Neighbouring reps share their top frame; the later one wins."""
    for rep in reversed(reps):
        if rep.start_frame <= frame_index <= rep.end_frame:
            return rep
    return None


def _pauses(faults: list[Fault]) -> dict[int, list[Fault]]:
    """The frames to pause on, each with the shown faults to list there.

    At most one pause per rep, at the moment of its last fault, listing all of them: two
    faults in one rep (a shallow bottom, then a sag on the way up) would otherwise freeze
    the video twice within a second. Pausing at the last one keeps every message after the
    fault it describes."""
    by_rep: dict[int, list[Fault]] = defaultdict(list)
    for fault in faults:
        if not fault.suppressed:
            by_rep[fault.rep_index].append(fault)
    pauses: dict[int, list[Fault]] = defaultdict(list)
    for rep_faults in by_rep.values():
        rep_faults.sort(key=lambda f: f.frame)
        pauses[rep_faults[-1].frame] += rep_faults
    return dict(pauses)


def _unchecked_summary(result: SessionResult, titles: dict[str, str]) -> list[str]:
    """One line per check that could not be made, in the user's words, with how many reps
    it affects: "Arm lockout: 8 of 8 reps"."""
    reps_by_rule: dict[str, int] = defaultdict(int)
    for unchecked in result.unevaluated:
        reps_by_rule[unchecked.rule] += 1
    total = len(result.reps)
    return [
        f"{titles.get(rule, rule)}: {count} of {total} reps" for rule, count in reps_by_rule.items()
    ]


def _draw_skeleton(
    frame: np.ndarray, points: np.ndarray, side: signals.Side, colour: tuple
) -> None:
    """Arm and body segments, the shoulder-ankle reference line, and the five joints."""
    scale = frame.shape[0] / 1080  # line widths that look the same at any resolution

    def xy(index: int) -> tuple[int, int] | None:
        x, y = points[index, :2]
        return None if np.isnan(x) or np.isnan(y) else (int(x), int(y))

    joints = [side.shoulder, side.elbow, side.wrist, side.hip, side.ankle]
    at = {index: xy(index) for index in joints}
    reference = (at[side.shoulder], at[side.ankle])
    if None not in reference:
        cv2.line(frame, *reference, NEUTRAL, max(1, round(2 * scale)), cv2.LINE_AA)
    for a, b in [
        (side.shoulder, side.elbow),
        (side.elbow, side.wrist),
        (side.shoulder, side.hip),
        (side.hip, side.ankle),
    ]:
        if at[a] is not None and at[b] is not None:
            cv2.line(frame, at[a], at[b], colour, max(2, round(6 * scale)), cv2.LINE_AA)
    for point in at.values():
        if point is not None:
            cv2.circle(frame, point, max(3, round(9 * scale)), colour, -1, cv2.LINE_AA)


def _draw_border(frame: np.ndarray, colour: tuple) -> None:
    """A frame around the picture, so a pause reads as deliberate, not as a stuck video."""
    thickness = max(4, round(12 * frame.shape[0] / 1080))
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), colour, thickness)


def _draw_panel(
    frame: np.ndarray, lines: list[tuple[str, tuple]], size: float = 1.1, middle: bool = False
) -> None:
    """Text lines on a dark box, centred across the frame, readable on any background. At
    the top by default; middle=True centres it vertically too, for the opening card."""
    if not lines:
        return
    scale = frame.shape[0] / 1080
    font_scale, thickness = size * scale, max(1, round(2 * size / 1.1 * scale))
    line_height = round(44 * size * scale)
    width = max(cv2.getTextSize(text, FONT, font_scale, thickness)[0][0] for text, _ in lines)
    pad = round(16 * size * scale)
    height = len(lines) * line_height + pad
    left = (frame.shape[1] - width) // 2 - pad
    top = (frame.shape[0] - height) // 2 if middle else 0
    cv2.rectangle(frame, (left, top), (left + width + 2 * pad, top + height), PANEL, -1)
    for row, (text, colour) in enumerate(lines, start=1):
        origin = (left + pad, top + row * line_height)
        cv2.putText(frame, text, origin, FONT, font_scale, colour, thickness, cv2.LINE_AA)
