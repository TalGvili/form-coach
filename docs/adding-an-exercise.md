# Adding an exercise

Form Coach analyses one exercise, the push-up. This page is the plan for a second one: what
already carries over, what has to change once before any second exercise fits, and the steps
for each new exercise after that. A squat filmed from the side is the running example.

No second exercise has been built yet, so this is a plan, written against the code as it is.
File and function names below are real.

## What carries over and what doesn't

The project was built so that the exercise lives in configuration. That is true for half the
pipeline.

| Part | Exercise-specific? | Why |
| --- | --- | --- |
| `landmarks.py` (MediaPipe, cache) | No | 33 body landmarks per frame, whatever the person is doing |
| `signals.py` building blocks: `angle`, `pick_side`, `runs`, `interpolate_gaps`, `smooth` | No | Geometry and signal cleaning |
| `reps.py` core: `find_rep_bottoms`, `rep_windows`, `completed_reps` | No | They need one signal that dips once per rep; any exercise has one |
| `rules.py`, `session.judge`, `progress.py` | No | They read metrics by name from the YAML |
| The web page's charts | No | Built from the `chart:` captions in the YAML |
| `signals.Side` | Partly | Has shoulder, elbow, wrist, hip, ankle; no knee |
| `signals.in_position_window` | **Yes** | "In position" means a horizontal body |
| `reps.clip_signals`, `reps.measure_rep` | **Yes** | Choose the upper-arm angle and hip deviation, compute push-up metrics |
| `models.Rep` | **Yes** | Its fields are push-up measurements (`max_hip_drop`, ...) |
| `render._draw_skeleton` | **Yes** | Draws the arm and body segments a push-up needs |
| `app/main.py`, `eval/evaluate.py`, `scripts/labels_from_md.py` | **Yes** | Each names `pushup.yaml`, its labels or its fault columns once |

The rule engine needs no changes. Measuring a new exercise needs new code, and that is
expected: knowing which joint angle shows squat depth is the exercise knowledge itself, and
it can't be written as a threshold.

## Part 1: before any code, the dataset

This is the slow part, and it can't be skipped. The evaluation numbers only mean something
because the labels were written down, filmed and set aside before the detector existed. The
order matters as much as the work.

1. **Write the labelling rule for each fault first**, from landmarks MediaPipe actually has.
   "The hip crease doesn't go below the top of the knee" is checkable with hip and knee points;
   "the lower back rounds" isn't, because there are no spine landmarks. Faults the camera
   angle can't show (knees caving in, from the side) are left out and listed as a limitation.
2. **Decide the borderline policy once**, before labelling, and write it in the sheet.
3. **Film.** The push-up set was 20 clips of 8–10 reps, each fault on its own, conditions varied
   on purpose (rooms, light, distance, which side faces the camera), the whole body in frame.
   Fewer clips are fine for a first version, as long as the README states the count: precision
   over 9 clips is weaker evidence than over 20, and reps within one clip aren't independent.
4. **Label one row per rep** in a Markdown sheet and generate the CSV from it, as
   `scripts/labels_from_md.py` does for push-ups. Its `FAULTS` list, sheet path and output path
   become per-exercise arguments.
5. **Choose the held-out clips now**, about 30%, so every fault appears in them at least once,
   and write the list down. They are not looked at again until tuning is finished.

## Part 2: once, before the second exercise

These changes make room for any number of exercises. After them, the push-up results must be
identical: re-run `python -m eval.evaluate` on the training clips and compare the tables
number for number. A refactor that changes a result is a bug, not a refactor.

### 2.1 Make `Rep` generic

`Rep` is a frozen dataclass with one field per push-up measurement. Three parts of the code
depend on that list of fields: `rules.load_rules` checks each rule's `metric:` against it,
`rules.py` and `render.py` read each metric's `<metric>_frame`, and `db.py` creates one column
of the `reps` table per field. A squat has different measurements, so there are two options.

**(a) One dataclass per exercise** (`PushupRep`, `SquatRep`). Every field stays typed, so a
misspelled metric is caught by the editor. But every consumer then takes a union of types, the
database needs a table per exercise and has to know which one to read, and each new exercise
adds a class, a table and a branch.

**(b) One `Rep` with the measurements in dictionaries.**

```python
@dataclass(frozen=True)
class Rep:
    index: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    metrics: dict[str, float]      # {"min_knee_angle": 82.4, ...}, NaN = could not measure
    metric_frames: dict[str, int]  # the frame each value came from
```

with a long table in place of one column per metric:

```sql
CREATE TABLE rep_metrics (
    session_id INTEGER NOT NULL,
    rep_index  INTEGER NOT NULL,
    name       TEXT NOT NULL,
    value      REAL,              -- NULL for NaN, as now
    frame      INTEGER NOT NULL,
    PRIMARY KEY (session_id, rep_index, name),
    FOREIGN KEY (session_id, rep_index) REFERENCES reps (session_id, "index") ON DELETE CASCADE
);
```

`rules.py` changes from `getattr(rep, rule.metric)` to `rep.metrics[rule.metric]`; it already
looks metrics up by name. A new metric is new rows, never a new column.

**The choice: (b).** Fixed fields suit data whose shape is known and stable, which was the
case with one exercise. Name/value pairs suit a set of names meant to grow. What (b) costs:

- A misspelled metric is a runtime `KeyError`, not an editor warning. The check moves to load
  time instead: each exercise declares the metric names it produces, and `load_rules` checks
  the YAML against that list, so a bad config still stops the server at startup.
- A dict inside a frozen dataclass can still be modified. Wrapping it in
  `types.MappingProxyType` makes it read-only.
- Existing databases need a migration: copy each push-up column into `rep_metrics` rows.

### 2.2 One module per exercise

The exercise-specific code in `reps.py` moves into `pipeline/exercises/pushup.py`, and a squat
gets `pipeline/exercises/squat.py`. Each provides the same three things:

- `METRICS`: the metric names it measures, for `load_rules` to check against.
- `clip_signals(landmarks, info, side, cfg)`: the in-position window and every series the
  reps are measured from, including the one depth signal segmentation runs on.
- `measure_rep(index, bottom, window, signals, fps, cfg) -> Rep`.

`segment_reps` keeps the shared steps (the no-person check, finding bottoms, dropping failed
reps) and calls the module the config names with `exercise:`. A dictionary from name to
module is enough; no plugin system.

`signals.Side` gets a `knee`. `pick_side` chooses the side by the mean visibility of
`Side.indices`, so adding the knee there would change which side is picked on some push-up
clips. Each exercise passes the joints it uses instead, and the push-up keeps its five.

### 2.3 The web app and history

- `app/main.py` holds one `CONFIG`, `RULES`, `TITLES` and `CHARTED`. They become a dictionary
  keyed by exercise, loaded at startup so a broken YAML still stops the server.
- `POST /analyze` takes an `exercise` form field, typed as a `Literal` of the known names so
  anything else is a 422. The page gets a picker; `NO_REPS_HINT` and the page's push-up wording
  move into each YAML.
- **`judged()` judges every stored session with the global `RULES`.** It must use the rules of
  `session.exercise`, or a squat session would be judged by push-up rules. The `sessions` table
  already stores `exercise`, so no schema change is needed for this.
- `/sessions` and `/progress` take an `exercise` filter: a chart mixing push-up depth with squat
  depth means nothing.
- `render._draw_skeleton` draws the joints and segments the exercise module lists.

### 2.4 Tests for the refactor

- The existing tests keep passing, with `Rep(...)` built through a small helper.
- A stored push-up session is still judged by the push-up rules after a squat config exists.
  This is the test that catches the `judged()` mistake above.
- `load_rules` refuses a rule whose metric the exercise doesn't produce.

## Part 3: for each new exercise

With Part 2 done, a new exercise is a labelled dataset, one module, one YAML file and its
tests.

1. **Choose the signals, and verify them before building on them.** For a squat: the knee angle
   (hip-knee-ankle) and the hip's height relative to the knee. Draw them on a few clips that
   answer a specific question. For any signed signal, plot it on a clip where the answer is
   known and on a mirrored clip, and add a synthetic test that asserts the sign. For push-ups
   the hip deviation's sign was the riskiest arithmetic in the project, because a wrong sign
   swaps two faults and nothing else would notice.
2. **Choose the depth signal to count reps on, by measurement.** For push-ups, three candidates
   were compared against labelled rep counts on 14 clips; an angle beat an absolute position
   because the body drifts in the frame. Do the same comparison, and log it in
   `eval/EXPERIMENTS.md`.
3. **Define "in position".** A squat set is done standing, so the window is the longest run
   where the torso is upright and the feet stay still, not the longest run where it's
   horizontal. Its thresholds go in the YAML's `segmentation:` section.
4. **Check rep counts against the labels before writing any rule.** A wrong count misaligns
   every rep after it with its labels. When a count disagrees, check the label as well as the
   code.
5. **Write `measure_rep`.** Return NaN for any metric whose landmarks were outside the frame,
   and record the frame each value came from.
6. **Write `configs/<exercise>.yaml`**: `exercise:`, `segmentation:`, and `rules:` with a
   `title`, `message` and a comment saying why each threshold has its value. Rule names match
   the label columns, so evaluation needs no mapping table.
7. **Tune on the training clips only**, one change at a time, each logged in
   `eval/EXPERIMENTS.md`. When two outcomes can't be separated by any threshold, document it as
   a limitation instead of tuning around it. Where it's a choice, lean toward missing a fault
   rather than flagging a good rep.
8. **Run the held-out clips once**, report that table in the README, and don't tune after it.
9. **Write the limitations**: what this camera angle and these landmarks can't measure.

## Checklist

- [ ] Labelling rules written before filming; borderline policy recorded
- [ ] Clips filmed with varied conditions; labels sheet and generated CSV committed
- [ ] Held-out clips chosen before any detection code
- [ ] Signals verified by plot and by synthetic tests, signs included
- [ ] Segmentation signal chosen by measurement, logged in `EXPERIMENTS.md`
- [ ] Rep counts match the labels on most training clips
- [ ] `pipeline/exercises/<exercise>.py` and `configs/<exercise>.yaml`; every threshold commented
- [ ] Training-clip tuning logged; held-out run once and reported
- [ ] Push-up results unchanged; `ruff check .` and `pytest` pass
- [ ] README: the exercise, its dataset, results and limitations
