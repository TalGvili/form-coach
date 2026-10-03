# Form Coach

[![CI](https://github.com/TalGvili/form-coach/actions/workflows/ci.yml/badge.svg)](https://github.com/TalGvili/form-coach/actions/workflows/ci.yml)

Web app that analyzes phone video of push-ups and gives per-rep form feedback: rep counting,
four fault types (shallow depth, sagging hips, piked hips, no lockout), and session history
to track progress over time.

Built with MediaPipe pose estimation, SciPy signal processing and a config-driven rule engine
behind a FastAPI service.

**Status:** in progress (Phase 4 — rule engine). Landmarks, signals and rep segmentation are
done; rep counts match the labels on 12 of the 14 training clips. Not yet usable.

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

20 side-view clips of push-ups by two people (person A in clip01–17, person B in
clip18–20), labeled one row per rep. `data/labeling_sheet.md` is the
hand-written record; `data/labels.csv` is generated from it by `scripts/labels_from_md.py`
and regenerated after every relabel, so the two never drift apart.

158 reps in total:

| Label | Reps | Clips |
| --- | --- | --- |
| `shallow` | 51 | 8 |
| `hip_sag` | 33 | 6 |
| `hip_pike` | 27 | 5 |
| `no_lockout` | 26 | 5 |
| clean (no fault) | 49 | 10 |

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

During tuning, three training labels were changed after re-watching the rep against the
written rule (clip14 rep 7, clip06 reps 5 and 7); each change and its reason is in
`eval/EXPERIMENTS.md`. All three moved toward the detector, so the training numbers are
somewhat optimistic. Held-out labels were never revisited.

### Fault isolation

Faults were also filmed on their own — piking and sagging with full depth, shallow reps with a
straight body — so each detector can be checked against its own fault rather than against faults
that happen to occur together. Reps carrying exactly one fault:

| Fault | Isolated reps |
| --- | --- |
| `no_lockout` | 26 / 26 |
| `hip_pike` | 20 / 27 |
| `shallow` | 23 / 51 |
| `hip_sag` | 12 / 33 |

`hip_sag` is the weakest here: 21 of its 33 reps also carry `shallow`, so only 12 reps across 2
clips show sag with depth otherwise correct. This is the known soft spot in the dataset.

### Held-out evaluation set

Six clips are held out and were chosen before any detection code was written:

**clip10** (sag) · **clip12** (clean, off-angle) · **clip15** (no lockout, mirrored) ·
**clip17** (pike) · **clip18** (shallow) · **clip20** (clean, mirrored)

49 of 158 reps, 31%. Every fault appears at least once. Thresholds and smoothing parameters are
tuned on the other 14 clips only; the final evaluation table is reported on these six, so the
numbers describe performance on video that was never used for tuning.

One cost is worth stating: clip10 holds 8 of the 12 isolated `hip_sag` reps, leaving 4 for
tuning. A test set with only one sag rep would have been worse.

Person B's clips split as one for tuning (clip19) and two held out (clip18, clip20), so the
held-out results are also reported per person: 20 reps from person B against 29 from person A.

## Limitations

What the method structurally cannot do, and why tuning won't fix it.

- **Two people, with the faults performed on purpose.** Body proportions, clothing, and the
  way a fault looks when it happens unintentionally all change the landmarks, and two people
  barely sample that. The second person appears in one tuning clip and two held-out clips, so
  the held-out results say something about a person the thresholds were barely tuned on, but
  nothing about people in general. Only filming more people can fix this; tuning can't.
- **A lower-back arch while the hips stay in line can't be measured.** The pose model has no
  landmarks along the spine, only shoulders and hips. Sag and pike are detectable because they
  move the hip landmark itself off the shoulder–ankle line; an arch does not.
- **Side view only.** Every measurement assumes the camera is roughly perpendicular to the body,
  at floor-to-hip height, with the whole body in frame. Front or angled views distort the joint
  angles the rules depend on.
- **Landmarks outside the frame make a metric unmeasurable, not wrong.** When the hands leave
  the frame, the pose model still reports a wrist position, extrapolated from the body with no
  warning. Any metric whose extreme value came from such a frame is reported as "could not
  evaluate" instead of a plausible guess. In three training clips the wrists are out of frame
  most of the time, so lockout can't be judged there; in one, the elbow also leaves the frame
  at the bottom of every rep, so depth can't be judged either.
- **Depth isn't reliable on reps with piked hips.** Detecting the pike itself works: it uses
  the hip's position against the shoulder–ankle line, which a side camera sees well. What
  suffers is the depth reading on those reps. With the hips piked, the elbows point toward
  the camera, so the upper arm is seen almost end-on: it shrinks on screen from ~125 to ~50
  pixels, and the angle of such a short segment stays steep even when the shoulder drops
  close to elbow height. A piked rep can therefore read as shallow when it isn't, and one of
  the 26 piked reps in the dataset isn't counted as a rep at all. Two alternatives were tried
  and rejected: a height-based depth signal, and a 3D angle from the pose model's own depth
  estimate. Both fixed the pike clip but miscounted three or four others. Measuring this
  properly needs a second camera or real 3D pose. In practice it matters less than it
  sounds: a piked rep at full depth is essentially a pike push-up, a harder variation, and
  was hard to perform even on purpose while filming.
- **A failed rep can look like a completed rep with bent arms.** A rep counts when the arms
  climb back most of the way relative to the clip's typical rep. A failed rep that gets halfway
  up before collapsing climbed back 0.64 of a typical rep; a completed rep without lockout in
  another clip, 0.63. The arm signal only records how far up the person got, not whether they
  meant to finish, so no threshold separates the two. For the same reason, collapsing flat onto
  the floor and then pushing up to straight arms looks exactly like a rep.
- **A single rep can't be analyzed.** A rep's start and end are cut at the clip's typical
  descent and ascent, learned from the gaps between reps; with one rep there are none, and the
  rep's edges can't be told apart from getting down and getting up. The pipeline reports this
  instead of guessing.
- **Failed reps are not analyzed.** They are excluded from the labels, so nothing measures them.
- **Head position and tempo are measurable but not implemented.** Both are stretch goals.
