"""Annotated video: what the analysis saw, drawn on every frame.

It draws only the five landmarks the pipeline measures (shoulder, elbow, wrist, hip, ankle
on the side facing the camera) and the shoulder-ankle line the hip is measured against, so
the viewer sees exactly what the numbers came from. During a rep the skeleton is green, or
red if the rep has a fault the feedback shows; fault messages are listed in red, and checks
that could not be evaluated in grey.

The output uses OpenCV's mp4v codec, which video players handle but browsers don't; the web
app re-encodes it to H.264 with ffmpeg.
"""

from pathlib import Path

import cv2
import numpy as np

from pipeline import signals
from pipeline.landmarks import load_or_extract
from pipeline.models import Rep, SessionResult

# OpenCV colours are BGR, not RGB.
OUTSIDE_REPS = (230, 230, 230)
CLEAN = (80, 200, 80)
FAULT = (60, 60, 230)
UNEVALUATED = (160, 160, 160)
PANEL = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX  # the built-in fonts draw ASCII only


def render(result: SessionResult, out: Path) -> None:
    """Write the annotated copy of result.video to out."""
    info, landmarks = load_or_extract(result.video.path)  # a cache hit after analyze()
    side = signals.pick_side(landmarks)
    capture = cv2.VideoCapture(info.path)
    size = (info.width, info.height)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, size)
    try:
        for frame_index in range(len(landmarks)):
            ok, frame = capture.read()
            if not ok:
                break
            rep = _rep_at(result.reps, frame_index)
            lines = _panel_lines(result, rep)
            colour = OUTSIDE_REPS if rep is None else CLEAN
            if any(c == FAULT for _, c in lines):
                colour = FAULT
            _draw_skeleton(frame, landmarks[frame_index], side, colour)
            _draw_panel(frame, lines)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def _rep_at(reps: list[Rep], frame_index: int) -> Rep | None:
    """The rep a frame belongs to. Neighbouring reps share their top frame; the later wins."""
    for rep in reversed(reps):
        if rep.start_frame <= frame_index <= rep.end_frame:
            return rep
    return None


def _panel_lines(result: SessionResult, rep: Rep | None) -> list[tuple[str, tuple]]:
    """The text for one frame: the rep number, then its shown faults, then unchecked rules."""
    if rep is None:
        return []
    lines: list[tuple[str, tuple]] = [(f"Rep {rep.index}", OUTSIDE_REPS)]
    for fault in result.faults:
        if fault.rep_index == rep.index and not fault.suppressed:
            lines.append((fault.message, FAULT))
    for unchecked in result.unevaluated:
        if unchecked.rep_index == rep.index:
            lines.append((f"Could not check: {unchecked.rule}", UNEVALUATED))
    return lines


def _draw_skeleton(frame: np.ndarray, points: np.ndarray, side: signals.Side, colour: tuple):
    """Arm and body segments, the shoulder-ankle reference line, and the five joints."""
    scale = frame.shape[0] / 1080  # line widths that look the same at any resolution

    def xy(index: int) -> tuple[int, int] | None:
        x, y = points[index, :2]
        return None if np.isnan(x) or np.isnan(y) else (int(x), int(y))

    joints = [side.shoulder, side.elbow, side.wrist, side.hip, side.ankle]
    at = {index: xy(index) for index in joints}
    reference = (at[side.shoulder], at[side.ankle])
    if None not in reference:
        cv2.line(frame, *reference, OUTSIDE_REPS, max(1, round(2 * scale)), cv2.LINE_AA)
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


def _draw_panel(frame: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    """Text lines on a dark box in the top-left corner, readable on any background."""
    if not lines:
        return
    scale = frame.shape[0] / 1080
    font_scale, thickness = 1.1 * scale, max(1, round(2 * scale))
    line_height = round(48 * scale)
    width = max(cv2.getTextSize(text, FONT, font_scale, thickness)[0][0] for text, _ in lines)
    pad = round(16 * scale)
    cv2.rectangle(frame, (0, 0), (width + 2 * pad, len(lines) * line_height + pad), PANEL, -1)
    for row, (text, colour) in enumerate(lines, start=1):
        origin = (pad, row * line_height)
        cv2.putText(frame, text, origin, FONT, font_scale, colour, thickness, cv2.LINE_AA)
