# Form Coach

Web app that analyzes phone video of push-ups and gives per-rep form feedback.
Full plan: @docs/form-coach-project-guide.md

## About me and how to help

- Third-year CS student; this is my first real project and portfolio piece for internship interviews.
- I must be able to explain every line in an interview. Default to explaining and reviewing,
  not writing whole modules for me. When I ask for code, keep it small and explain the why.
- Point out bugs, bad design and missing tests directly.

## Current status

- Phase: 4 (rule engine and feedback)
- Done: `models.py`; `landmarks.py` with the `.npz` cache (all 20 clips cached);
  `signals.py` and `reps.py` with tests. The hip-deviation sign is verified on real clips
  and by a mirror test. Rep counts match the labels on 12 of 14 training clips (clip08:
  failed rep reads like a no-lockout rep; clip13: a pike rep's dip is too small).
- Next: `pipeline/rules.py` and the `rules:` section of `configs/pushup.yaml`, with the
  three outcomes: fault, no fault, could not evaluate (NaN). Undecided: whether `hip_pike`
  suppresses `shallow` in the feedback (guide, Phase 4 Step 4).
- Metric choices: `shallow` uses `min_upper_arm_angle`, `no_lockout` uses `max_elbow_angle`,
  hip faults use the signed hip deviation. The measurements behind these are in
  `docs/notes.txt` (private, gitignored).

Phase 1 is done: 20 clips, 158 labeled reps in `data/labels.csv`, generated from
`data/labeling_sheet.md`. Held-out clips (never used for tuning): 10, 12, 15, 17, 18, 20.

## Conventions

- Python 3.12 in `.venv` (MediaPipe is only officially supported up to 3.12).
- MediaPipe Tasks API (`PoseLandmarker`), never the old `mp.solutions`.
- Core types are dataclasses in `pipeline/models.py`; landmarks are NumPy arrays `(n_frames, 33, 4)`.
- Pipeline code never imports anything web-related; `app/` never contains analysis logic.
- All thresholds live in `configs/*.yaml`, no magic numbers in code.
- Tests use synthetic signals, never the private videos in `data/`.
- Run `ruff check .` and `pytest` before every commit.
