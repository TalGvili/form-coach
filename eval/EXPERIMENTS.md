# Experiments

Every change made while tuning, with the numbers before and after and what I concluded. Measured
on the 14 training clips only; the held-out clips (10, 12, 15, 17, 18, 20) are not used until
tuning is finished.

Reps within one clip share a camera, a room and a body, so they are not independent samples:
"5 false alarms" from one clip is weaker evidence than 5 spread over five clips. The per-clip
breakdown (`python -m eval.evaluate --by-clip`) is there to tell the two apart.

Experiments 1–6 were run during Phase 3, before `eval/evaluate.py` existed, with one-off scripts
over the same clips. Where they report fault counts, they used the planned rules
(15° / 0.3 s for the hips) on whole-clip counts, not the evaluation table below.

## 1. Which signal to segment reps on (Phase 3)

**Change:** count reps on three candidate signals with one untuned parameter set, and compare
with the labelled counts.

| Signal | Clips counted correctly |
| --- | --- |
| Upper-arm angle | 9 of 14 |
| Elbow angle | 8 of 14 |
| Shoulder height (pixels) | 7 of 14 |

**Conclusion:** segment on the upper-arm angle. Shoulder height failed worst (1–2 peaks where 8
were expected): it is an absolute position, so the body drifting in the frame swamps the
movement. Angles are relative geometry and don't move with the body.

## 2. Bounding the first and last rep (2026-09-27)

**Problem:** the first and last rep have no neighbour on one side, so their window ran to the
edge of the in-position window and took in getting down and getting up. Hip rise on edge reps
read 33–103° against 7–13° for the others: hip_pike would have false-alarmed on nearly every
clip's first and last rep (~20 false alarms, against 26 real pikes in the dataset).

**Change:** cut the open side at the clip's median descent / ascent length.

**Result:** e.g. clip01's first rep, hip rise 103° → 1°. Some inflation remained (see 3).

## 3. Measure hips only while the arms move (2026-09-28)

**Problem:** the remaining edge-rep spikes all sat at the rep's outer frame, with the arms
already at the top: the hip moves while the person settles into the plank or starts getting up.

**Change:** hip metrics use only frames where the upper arm is below a fraction of the way from
the rep's bottom to its top (`hip_moving_fraction`).

| Fraction | Pike: hit / miss / false | Sag: hit / miss / false |
| --- | --- | --- |
| 1.0 (all frames) | 11 / 0 / 12 | 17 / 5 / 1 |
| **0.9** | 11 / 0 / 5 | 17 / 5 / 0 |
| 0.8 | 11 / 0 / 4 | 16 / 6 / 0 |

**Conclusion:** 0.9. It removes 8 false alarms and loses nothing; 0.8 starts losing real sags.
It also matches the labelling rule ("during the movement, not while pausing at the top").

## 4. Failed reps that look completed (2026-09-28)

**Problem:** clip11 counted 6 reps for 5. After the failed rep, pushing off the floor made a
dip of its own. That dip was rejected, but while it existed it set the "top" ending the failed
rep, and that top was the getting up, so the failed rep looked complete.

**Tried first:** count a rep's rise only within the clip's typical ascent time. Broke clip03
and clip14 (real reps that pause on the way up) and didn't fix clip11. Rejected.

**Change:** a rejected bottom is not a rep, so it can't decide where its neighbour ends. Drop
it and recompute the windows until every remaining rep completes.

**Result:** clip11 7 → 6 → 5 bottoms, correct. No other clip changed. Counts: 12 of 14.

**Not fixable this way:** clip08's failed rep climbs back 0.64 of a typical rep; clip16 rep 2,
a completed rep without lockout, 0.63. No threshold separates them.

## 5. Other depth signals for pike reps (2026-09-28 / 29)

**Problem:** clip13 misses a pike rep, and pike reps read 54–73° "deep" when the shoulder is
close to elbow height. With the hips piked the elbows point at the camera, and the upper arm,
seen end-on, keeps a steep angle.

| Depth signal | clip13 | Other clips counted correctly | Shallow reps inside the clean range |
| --- | --- | --- | --- |
| Upper-arm angle (current) | 7 of 8 | 12 of 13 | 1 |
| Shoulder height above elbow / arm length | 8 of 8 | 9 of 13 | 9 |
| 3D angle, using MediaPipe's depth estimate | 8 of 8 | 8 of 13 | 5 |

**Conclusion:** both alternatives fix the pike clip and break three or four others. Kept the
2D angle; the limitation is in the README.

## 6. Jitter counted as reps (2026-09-29)

**Found by a test:** a clip with a plank held still reported reps. The prominence threshold is a
share of the clip's own range, and with no reps the range is only jitter.

**Change:** a clip whose upper-arm range stays below 10° has no reps (`min_range_deg`). The
smallest range with real reps is 15° (clip03); a held pause spans ~6°. A minimum dip size in
degrees was rejected: clip03's real reps dip 6°, its wobbles 4°.

**Result:** rep counts unchanged.

## 7. Baseline (2026-10-02)

First run of `python -m eval.evaluate`. Starting thresholds: shallow 5°, hips 15° for 0.3 s,
lockout 160°. 12 of 14 clips counted correctly; clip08 and clip13 are left out of the per-rep
scoring. **94 reps from 12 clips.**

| Fault | Labelled | Detected | Missed | False alarms | Precision | Recall | Not evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shallow | 35 | 35 | 0 | 18 | 0.66 | 1.00 | 8 |
| hip_sag | 22 | 17 | 5 | 0 | 1.00 | 0.77 | 0 |
| hip_pike | 16 | 16 | 0 | 5 | 0.76 | 1.00 | 3 |
| no_lockout | 13 | 13 | 0 | 4 | 0.76 | 1.00 | 22 |

Where the errors are:

- **shallow, 18 false alarms over 6 clips** (7 in clip07). Clean reps read just above 5°. In
  clip16 the raw signal reached 0.1–2.6° at every bottom; smoothing lifts narrow dips by 1–5°.
- **hip_sag, 5 misses, all in clip04 and clip05.** Short sags under the 0.3 s minimum.
- **hip_pike, 5 false alarms**, 2 in clip06, whose hip signal looks offset over the whole clip.
- **no_lockout, 4 false alarms, 3 in clip01**, clean reps at 158–162°.
- **no_lockout not evaluated on 22 reps**: clips 03, 04 and 05 have the wrists out of frame.

**Suppression trade-off.** hip_pike suppresses shallow in the feedback. On these clips:

| shallow | Detected | Missed | False alarms | Recall |
| --- | --- | --- | --- | --- |
| Detector | 35 | 0 | 18 | 1.00 |
| Feedback | 28 | 7 | 14 | 0.80 |

It hides 4 false alarms and 7 correct detections, mostly clip03, whose reps are labelled both
piked and shallow and where the depth reading was right. The reason given for suppression,
"depth reads wrong on piked reps", held for clip13 but not for clip03. Kept for now on the
coaching argument (fix the hips first); revisit once shallow's false alarms are fixed.

**Next:** shallow first (most false alarms, known cause).

## 8. Shallow: read depth unsmoothed, check the labels, move the threshold (2026-10-03)

**Step 1, read the bottom from the raw signal.** Bottoms are found on the smoothed signal, which
rounds off the tip of a narrow dip: in clip16 the raw upper-arm angle reached 0.1–2.6° at
every bottom, the smoothed one 2.5–6.8°, and rep 1 (raw 2.3°, smoothed 6.5°) was flagged.
Depth is now read from the gap-filled but unsmoothed signal at the bottom frame.

| Depth read from | False alarms | Missed |
| --- | --- | --- |
| Smoothed signal (baseline) | 18 | 0 |
| Raw, at the bottom frame | 13 | 0 |
| Raw, lowest within ±0.05 / ±0.1 / ±0.2 s | 14 / 14 / 13 | 0 |

Searching around the bottom doesn't help, so there is no search window and no setting for it.
Smoothing explained only 5 of the 18.

**Step 2, look at the reps that still fail.** With the raw reading, clean-labelled reps read up
to 17.3° and the shallowest shallow-labelled rep 18.9°. Drawing the bottoms showed full-depth
reps (face near the floor) whose shoulder sits only slightly above the elbow: the elbows flare
toward the camera, the upper arm is seen partly end-on, and a small height difference over a
short segment reads as a steep angle. The same effect as the pike limitation, milder. A 5°
threshold assumes a perfect side view.

The same pictures showed one label to be wrong: clip14 rep 7 (16.1°, already noted "maybe
shallow" when labelling) is shallow on re-watching, and is relabelled. `shallow` is now 51 reps
in 8 clips.

**Step 3, the threshold.** Readings now: normal clean reps up to 13.8° (clip07 rep 5), two
piked clean reps at 15.0° and 17.3° (clip11; piked depth misreads, and the feedback suppresses
it), shallow reps from 16.1°.

| Threshold | Detector false alarms | Feedback false alarms | Missed | Room below 16.1° | Room above 13.8° |
| --- | --- | --- | --- | --- | --- |
| 5 | 12 | 8 | 0 | 11.1° | – |
| 12 | 5 | 2 | 0 | 4.1° | – |
| 14 | 2 | 0 | 0 | 2.1° | 0.2° |
| **15** | 1 | 0 | 0 | 1.1° | 1.2° |

Both 12 and 14 miss nothing here; they differ in which mistake they risk on new video. 12 keeps
4° of room before a shallow rep is missed and accepts 2 false alarms the user sees. 14 shows no
false alarms on these clips but sits 0.2° above the highest clean rep and 2.1° below the
shallowest shallow one, and that boundary is set by a single borderline rep.

The gap between the groups (2.3°) is smaller than how much readings shift between videos, so
on new video some reps will land on the wrong side wherever the line is. The threshold only
decides which mistake comes first.

**Decision: 15.** For a user, a false "not deep enough" on a good rep is worse than silence on
a shallow one: it is annoying and makes the rest of the feedback less trusted. 15 is the middle
of the gap, tilted toward missing: 1.2° of room before a clean rep is flagged (14 had 0.2°) and
1.1° before a shallow rep is missed. The one remaining detector false alarm is a piked rep
(clip11 rep 3), which suppression hides from the feedback. Fixed before running the held-out
clips, and not to be changed after.

## 9. Hip sag: separate thresholds, 13° for 0.2 s (2026-10-03)

**Problem:** 5 labelled sags missed, all in clip04 (reps 6–8) and clip05 (reps 6–7). The
duration rule was not the main cause: these sags barely reach 15° at all.

| | Hip drop |
| --- | --- |
| Missed sags | 12.5, 12.6, 13.7, 15.1 (for 0.03 s), 15.4° (for 0.07 s) |
| Caught sags | from 17.2° |
| Clean reps, highest | 12.6° (clip16 rep 2), then 11.6, 11.6, 11.4 |

**First try:** lower the shared hip threshold. Sag improved, but pike false alarms rose from
5 to 7–12: sag and pike shared one number (`hip_duration_threshold_deg`), because the
durations were measured at a single threshold.

**Change:** separate `sag_threshold_deg` and `pike_threshold_deg`, each defined once and
anchored to its own rule. Pike stays at 15° / 0.3 s and its numbers don't move in any
variant below.

| Sag threshold | Min. duration | Caught (of 22) | Missed | False alarms |
| --- | --- | --- | --- | --- |
| 15° | 0.3 s | 17 | 5 | 0 |
| 14° | 0.2 s | 18 | 4 | 0 |
| 13° | 0.3 s | 19 | 3 | 0 |
| **13°** | **0.2 s** | **20** | **2** | **0** |
| 12° | 0.2 s | 21 | 1 | 1 |
| 12° | 0.15 s | 22 | 0 | 1 |

The duration rule is what lets the threshold drop: clean clip16 reps touch 12° but don't stay
there. The one false alarm at 12° is clip16 rep 2 (12.6° for 0.23 s). Drawn at its worst
moment next to clip05 rep 6, a labelled sag with the same 12.6°, the two look alike: the hip
visibly below the shoulder–ankle line in both. Re-watched, both are borderline, so the labels
stay as they are.

**Decision: 13° for 0.2 s.** Catches 3 more sags with no false alarms, by the same rule as no.
8 (a miss is better than a false alarm). The 2 still missed are as small as clean reps get.

**Caveats:** the gain comes almost entirely from clip04's later reps, whose sags were smaller,
so the held-out set may show less. And the overlap at 12.5° points to a per-clip offset: some
clips read 10–12° below the line on every rep (clip16 for sag; clip06 does the same for pike).
Measuring each rep's hip relative to the person's own plank could remove it; a separate
experiment.

## 10. No lockout: one relabel, threshold 156° (2026-10-03)

**Problem:** 4 false alarms at the starting minimum of 160°, 3 of them in clip01. The two
groups touched: labelled no-lockout reps read up to 155.0°, the most bent locked-out rep
155.4° (clip06 rep 7).

**Looking at the boundary.** The top of four reps, drawn with a line continuing the upper arm
straight: clip07 rep 7 (no lockout, 155.0°), clip06 rep 6 (no lockout, 153.4°), clip06 rep 7
(locked, 155.4°), clip01 rep 2 (locked, 157.5°). All four show a slight bend of about the same
size; even reps labelled locked out read 155–158°, not 175–180°. Re-watched, clip06 rep 7 is
no lockout, by a small amount, like rep 6 next to it, and is relabelled. `no_lockout` is now
26 reps.

**After the relabel** the groups separate: no lockout up to 155.4°, locked out from 157.5°.

| Minimum | False alarms | Missed |
| --- | --- | --- |
| 160 (start) | 3 | 0 |
| 158 | 1 | 0 |
| **156** | **0** | **0** |
| 155 | 0 | 2 |

**Decision: 156°.** The middle of the 2.1° gap is 156.4°; 156 nudges it toward missing, by
the rule from no. 8. Room: 1.5° before the nearest locked-out rep is flagged, 0.6° before a
no-lockout rep is missed. 14 of 14 detected, no false alarms. The gap is narrow and set by
borderline reps, so the held-out set may cost a miss or a false alarm.

**Not evaluated:** lockout can't be checked on 22 of the 94 scored reps (clips 03, 04, 05,
wrists outside the frame). No threshold changes that.
