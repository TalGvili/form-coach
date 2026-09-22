# Form Coach

Web app that analyzes phone video of push-ups and gives per-rep form feedback: rep counting,
four fault types (shallow depth, sagging hips, piked hips, no lockout), and session history
to track progress over time.

Built with MediaPipe pose estimation, SciPy signal processing and a config-driven rule engine
behind a FastAPI service.

**Status:** in progress (Phase 1 — filming and labelling the dataset). Not yet usable.

## Limitations

- **Lower-back arch is not detected.** MediaPipe's pose model has no landmarks along the spine,
  only shoulders and hips, so an arched back with the hips still on the shoulder–ankle line is
  invisible to this method. Hip sag and pike are detectable because they move the hip landmark
  itself off that line.
- **Side view only.** Every measurement assumes the camera is roughly perpendicular to the body,
  at floor-to-hip height, with the whole body in frame. Front or angled views distort the joint
  angles the rules depend on.
- **Head position and tempo are measurable but not implemented.** Both are stretch goals.
