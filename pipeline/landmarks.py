"""MediaPipe PoseLandmarker extraction and the landmark cache.

MediaPipe is the slow step in the pipeline: minutes per video, against seconds for
everything downstream. The cache holds the landmarks and the video metadata together so
later phases can re-run without the original video, which is what makes threshold tuning
in Phase 5 practical.
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
    RunningMode,
)

from pipeline.models import VideoInfo

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
CACHE_DIR = ROOT / "data" / "cache"

N_LANDMARKS = 33  # the pose model always returns this many points
N_COORDS = 4  # x pixels, y pixels, z, visibility


def cache_path(video_path: str | Path) -> Path:
    """Where the landmarks for this video are cached."""
    return CACHE_DIR / f"{Path(video_path).stem}.npz"


def _frame_to_row(
    result: PoseLandmarkerResult, width: int, height: int
) -> np.ndarray:  # converts one frame's media pipe result into a (33,4) block.
    """One frame's landmarks as (33, 4), all NaN when no person was detected."""
    row = np.full((N_LANDMARKS, N_COORDS), np.nan)  # create a 33x4 array, all values NaN
    if not result.pose_landmarks:
        return row

    # num_poses defaults to 1, so there is at most one pose to read.
    for i, landmark in enumerate(result.pose_landmarks[0]):
        # Normalized coords are relative to width and height separately. At 1920x1080
        # one x unit spans 1.78 times as many pixels as one y unit, so angles computed
        # before this conversion are wrong.
        row[i] = (landmark.x * width, landmark.y * height, landmark.z, landmark.visibility)
    return row


def extract_landmarks(path: str | Path) -> tuple[VideoInfo, np.ndarray]:
    """Run MediaPipe over every frame of a video.

    Returns the video metadata and an array of shape (n_frames, 33, 4) holding x pixels,
    y pixels, z and visibility. Frames with no detection stay NaN rather than being
    dropped, so row i is always frame i and Phase 3 can interpolate short gaps.
    """
    path = Path(path)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose model not found at {MODEL_PATH}. It is a 9 MB binary kept out of git; "
            "see the Setup section of the README for the download command."
        )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
    )

    rows: list[np.ndarray] = []
    last_timestamp_ms = -1
    # Both objects hold native resources: the video file handle, and the landmarker's
    # model plus its per-video tracking state. Closing them is not optional, so both get
    # a definite lifetime rather than being left to the garbage collector.
    with (
        cv2.VideoCapture(str(path)) as capture,
        PoseLandmarker.create_from_options(options) as landmarker,
    ):
        if not capture.isOpened():
            # OpenCV hands back an unopened object instead of raising, so check it.
            raise OSError(f"Could not open video: {path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            # OpenCV decodes to BGR, MediaPipe expects RGB. Detection still runs on
            # BGR, just worse, so a missing conversion looks like a bad model.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # VIDEO mode rejects a timestamp that does not advance. Integer
            # truncation can repeat one at high frame rates, so clamp upwards.
            timestamp_ms = max(int(len(rows) * 1000 / fps), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(image, timestamp_ms)
            rows.append(_frame_to_row(result, width, height))

    # Built from the frames actually read, not from CAP_PROP_FRAME_COUNT: that value
    # comes from container metadata and can disagree with the real frame count.
    landmarks = np.stack(rows) if rows else np.empty((0, N_LANDMARKS, N_COORDS))
    info = VideoInfo(
        path=str(path),
        fps=fps,
        width=width,
        height=height,
        n_frames=len(landmarks),
    )
    return info, landmarks


def load_or_extract(path: str | Path) -> tuple[VideoInfo, np.ndarray]:
    """Return cached landmarks for a video, extracting and caching them on a miss."""
    cache_file = cache_path(path)
    if cache_file.exists():
        with np.load(cache_file) as data:
            landmarks = data["landmarks"]
            info = VideoInfo(
                path=str(path),
                fps=float(data["fps"]),
                width=int(data["width"]),
                height=int(data["height"]),
                n_frames=len(landmarks),
            )
        return info, landmarks

    info, landmarks = extract_landmarks(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        landmarks=landmarks,
        fps=info.fps,
        width=info.width,
        height=info.height,
    )
    return info, landmarks


def count_undetected(landmarks: np.ndarray) -> int:
    """Frames where no person was found, i.e. the whole row is NaN."""
    return int(np.isnan(landmarks).all(axis=(1, 2)).sum())


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m pipeline.landmarks <video>", file=sys.stderr)
        return 2

    info, landmarks = load_or_extract(argv[0])
    undetected = count_undetected(landmarks)
    print(info.path)
    print(f"  {info.width}x{info.height} at {info.fps:.2f} fps")
    print(f"  {info.n_frames} frames, {undetected} with no person detected")
    print(f"  landmarks {landmarks.shape}, cached at {cache_path(argv[0]).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
