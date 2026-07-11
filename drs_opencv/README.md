# DRS (Decision Review System) — OpenCV Simulation

A self-contained Python + OpenCV project that simulates the core pipeline
behind cricket's ball-tracking DRS: **detect → track → predict → decide**,
and renders a broadcast-style review graphic (Pitching / Impact / Wickets /
final call), plus an annotated tracking video.

> ⚠️ **This is an educational simulation, not a real/certified DRS.**
> Official systems (e.g. Hawk-Eye) use multiple synchronized, calibrated
> cameras and full 3D physics (including ball height, swing, spin, and
> bounce dynamics) to triangulate the ball in real space. This project
> uses a **single 2D camera view** and a simplified lateral-only model —
> see [Limitations](#limitations) below.

## What it does

1. **Detects** the ball in each video frame using HSV colour thresholding
   + contour circularity checks, with a Hough-circle fallback for tricky
   frames.
2. **Tracks** the ball across frames with a Kalman filter, smoothing noisy
   detections and coasting through brief occlusions (e.g. bat/pad).
3. **Predicts** the trajectory: fits a piecewise-linear model to find the
   pitching (bounce) point, then projects the pre-impact line of the ball
   forward to the stumps to see where it *would* have gone.
4. **Classifies** the delivery the way broadcasters do:
   - `PITCHING`: Outside Leg / In-line / Outside Off
   - `IMPACT`: Outside Leg / In-line / Outside Off
   - `WICKETS`: Hitting / Umpire's Call / Missing
   - Combines these into a final **OUT / NOT OUT / UMPIRE'S CALL**
5. **Renders** an annotated tracking video and a final decision-graphic
   image (green pitch map, dotted predicted path, colour-coded stumps).

## Project structure

```
drs_opencv/
├── config.py                 # all tunable constants (colors, geometry, thresholds)
├── ball_detector.py          # HSV + contour + Hough ball detection
├── tracker.py                # Kalman-filter based multi-frame tracking
├── trajectory_predictor.py   # pitch-point fitting + forward projection
├── stump_zone.py             # stump geometry + zone classification logic
├── visualizer.py             # drawing: live overlay + final decision graphic
├── generate_test_video.py    # creates synthetic test footage (no real video needed)
├── main.py                   # CLI entry point, orchestrates the full pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

(Python 3.9+ recommended.)

## Quick start (no video needed)

The project ships with a synthetic-footage generator, so you can try the
whole pipeline immediately without any real cricket video:

```bash
# Generate a test delivery where the ball ends up hitting the stumps
python generate_test_video.py --out sample_input.mp4 --outcome hitting

# Run the full DRS pipeline on it
python main.py --input sample_input.mp4 --output_dir output
```

Other synthetic outcomes to try:

```bash
python generate_test_video.py --out sample_missing.mp4 --outcome missing
python generate_test_video.py --out sample_umpires_call.mp4 --outcome umpires_call
```

Outputs land in `output/`:
- `tracked_output.mp4` — original video with live ball tracking overlay
- `drs_decision.png` — final broadcast-style review graphic

## Using your own footage

```bash
python main.py --input your_video.mp4 --output_dir output --color red
# or, for a white ball:
python main.py --input your_video.mp4 --output_dir output --color white
```

Real footage will very likely need threshold tuning — cricket balls vary
in exact colour, lighting differs by ground/broadcast, and camera angle
affects the pitch/stump geometry. Start by adjusting these in `config.py`:

- `RED_BALL_LOWER_1/2`, `RED_BALL_UPPER_1/2` (or `WHITE_BALL_*`) — HSV
  colour range for your ball. Use a colour-picker tool on a still frame
  to sample the actual ball pixels.
- `MIN_BALL_RADIUS` / `MAX_BALL_RADIUS` — expected ball size in pixels at
  your camera's resolution/distance.
- `BOWLER_END_Y`, `BATSMAN_END_Y`, `STUMPS_WIDTH_FAR/NEAR`,
  `STUMPS_HEIGHT_FAR/NEAR` — pixel geometry of the pitch/stumps as seen
  in your specific camera framing.

## How the "pitch point" is found

Rather than assuming a fixed bounce frame, `trajectory_predictor.py` runs
a small piecewise-linear regression search: it tries every plausible
split point along the tracked path, fits a straight line to the points
before and after each candidate split, and keeps the split that minimizes
total fitting error. This works because a delivery's lateral position is
close to linear before the bounce and (potentially) a *different* linear
line after the bounce (seam movement changing the angle) — so the best
two-line fit naturally lands on the bounce point.

## Limitations

This project is intentionally scoped as a learning/demo tool, not a
production system:

- **Single camera, 2D only.** No height/depth triangulation — a real DRS
  reasons in 3D (would the ball have gone over the stumps?), this project
  only reasons about lateral (left/right) position.
- **No handedness detection.** Off-side vs leg-side is fixed to +x/-x in
  the frame; flip this logic yourself if tracking a left-handed batsman
  or a mirrored camera angle.
- **Straight-line post-impact projection.** Real Hawk-Eye continues the
  ball's *physically modeled* path (accounting for further potential
  seam/spin) — this project uses the simpler assumption that the
  pre-impact line continues unchanged, which is a reasonable
  approximation for many deliveries but not physically exhaustive.
- **Colour-based detection.** Works well in controlled/synthetic footage;
  real broadcast footage (motion blur, players wearing similar colours,
  variable lighting) will need threshold tuning and would benefit from a
  trained object detector for production use.

## Extending this project

- Swap `ball_detector.py`'s colour-threshold approach for a small trained
  model (e.g. YOLO fine-tuned on cricket-ball footage) for real-world
  robustness.
- Add a second camera feed and triangulate for true 3D tracking (this is
  the biggest step toward a "real" Hawk-Eye-style system).
- Add height/vertical modeling using a parabolic fit (gravity-based) to
  reason about whether the ball would have cleared the stumps.
- Add a simple GUI (e.g. Streamlit) to upload a video and view results
  interactively instead of via CLI.
