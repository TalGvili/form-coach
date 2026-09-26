# Form Coach

Web app that analyzes phone video of push-ups and gives per-rep form feedback: rep counting,
four fault types (shallow depth, sagging hips, piked hips, no lockout), and session history
to track progress over time.

Built with MediaPipe pose estimation, SciPy signal processing and a config-driven rule engine
behind a FastAPI service.

**Status:** in progress (Phase 2 — landmark extraction). Not yet usable.

## Setup

Python 3.12 — MediaPipe does not publish wheels for newer versions.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pose model is a 9 MB binary and is not tracked in git, so download it separately:

```bash
mkdir -p models
curl -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

The push-up clips are private and not in the repo either, so the pipeline can't be re-run on
the original dataset. `data/labels.csv` and `data/labeling_sheet.md` are included, so the
labels and the evaluation method can still be inspected.

## Dataset and labeling

20 side-view clips of push-ups, labeled one row per rep. `data/labeling_sheet.md` is the
hand-written record; `data/labels.csv` is generated from it by `scripts/labels_from_md.py`
and regenerated after every relabel, so the two never drift apart.

158 reps in total:

| Label | Reps | Clips |
| --- | --- | --- |
| `shallow` | 50 | 7 |
| `hip_sag` | 33 | 6 |
| `hip_pike` | 26 | 5 |
| `no_lockout` | 24 | 4 |
| clean (no fault) | 52 | 10 |

A rep can carry more than one fault, so the fault counts overlap and do not sum to 158. The
clip count matters as much as the rep count: reps within one clip share lighting, camera
position and body proportions, so they are not independent samples.

### Labeling rules

Label what is visible in the video, not what the rep was meant to be.

| Column | Label 1 when… |
| --- | --- |
| `shallow` | At the bottom, the shoulder doesn't drop to elbow height (the upper arm never reaches parallel to the floor) |
| `hip_sag` | The hips visibly drop below the shoulder–ankle line for a noticeable part of the rep. Counts only during the movement itself (the descent and the push back up), not while pausing at the top between reps |
| `hip_pike` | The hips visibly rise above the shoulder–ankle line (the body forms an inverted V) |
| `no_lockout` | At the top, the arms stop visibly short of straight. A clear bend that should be corrected, not a slight softness in the elbows |

A rep counts only when the person returns to the top position. Failed final reps are excluded
from the labels; there are 3 in this dataset.

Borderline reps are labeled using the rule anyway, with a note in the `notes` column.

### Fault isolation

Faults were also filmed on their own — piking and sagging with full depth, shallow reps with a
straight body — so each detector can be checked against its own fault rather than against faults
that happen to occur together. Reps carrying exactly one fault:

| Fault | Isolated reps |
| --- | --- |
| `no_lockout` | 24 / 24 |
| `hip_pike` | 19 / 26 |
| `shallow` | 23 / 50 |
| `hip_sag` | 13 / 33 |

`hip_sag` is the weakest here: 20 of its 33 reps also carry `shallow`, so only 13 reps across 3
clips show sag with depth otherwise correct. This is the known soft spot in the dataset.

### Held-out evaluation set

Six clips are held out and were chosen before any detection code was written:

**clip10** (sag) · **clip12** (clean, off-angle) · **clip15** (no lockout, mirrored) ·
**clip17** (pike) · **clip18** (shallow) · **clip20** (clean, mirrored)

49 of 158 reps, 31%. Every fault appears at least once. Thresholds and smoothing parameters are
tuned on the other 14 clips only; the final evaluation table is reported on these six, so the
numbers describe performance on video that was never used for tuning.

One cost is worth stating: clip10 holds 8 of the 13 isolated `hip_sag` reps, leaving 5 for
tuning. A test set with only one sag rep would have been worse.

### Limitations

- **A lower-back arch while the hips stay in line can't be measured.** The pose model has no
  landmarks along the spine, only shoulders and hips. Sag and pike are detectable because they
  move the hip landmark itself off the shoulder–ankle line; an arch does not.
- **Failed reps are not analyzed.** They are excluded from the labels, so nothing measures them.
- **Side view only.** Every measurement assumes the camera is roughly perpendicular to the body,
  at floor-to-hip height, with the whole body in frame. Front or angled views distort the joint
  angles the rules depend on.
- **Head position and tempo are measurable but not implemented.** Both are stretch goals.
