# Form Coach

Web app that analyzes phone video of push-ups and gives per-rep form feedback.
Full plan: @docs/form-coach-project-guide.md

## About me and how to help

- Third-year CS student; this is my first real project and portfolio piece for internship interviews.
- I must be able to explain every line in an interview. Default to explaining and reviewing,
  not writing whole modules for me. When I ask for code, keep it small and explain the why.
- Point out bugs, bad design and missing tests directly.

## Current status

- Phase: 1 (film and label the dataset)
- Next: film 15-20 side-view clips into `data/videos/`, then fill `data/labels.csv` one row per rep.

## Conventions

- Python 3.12 in `.venv` (MediaPipe is only officially supported up to 3.12).
- MediaPipe Tasks API (`PoseLandmarker`), never the old `mp.solutions`.
- Core types are dataclasses in `pipeline/models.py`; landmarks are NumPy arrays `(n_frames, 33, 4)`.
- Pipeline code never imports anything web-related; `app/` never contains analysis logic.
- All thresholds live in `configs/*.yaml`, no magic numbers in code.
- Tests use synthetic signals, never the private videos in `data/`.
- Run `ruff check .` and `pytest` before every commit.
