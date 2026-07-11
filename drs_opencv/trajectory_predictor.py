"""
trajectory_predictor.py
------------------------
Takes the tracked ball trajectory (list of x,y points, where y represents
depth from bowler's end -> batsman's end and x represents lateral
position) and works out:

  1. The pitching point -- where the ball bounces / changes direction.
  2. A predicted straight-line continuation of the pre-impact trajectory,
     projected forward to the stumps -- this is the simplified stand-in
     for what real Hawk-Eye systems compute with full 3D physics.

Approach for finding the pitching point: a delivery's line is very close
to piecewise-linear in this simplified 2D (x=lateral, y=depth) model --
one segment before the bounce, one after (since seam movement/swing
changes the angle at the bounce). We do a small piecewise-linear
regression search: try every plausible split index, fit a line to each
side, and keep the split with the lowest combined residual error.
"""

import numpy as np
import config as cfg


class TrajectoryPrediction:
    def __init__(self):
        self.has_prediction = False
        self.pitch_point = None       # (x, y)
        self.impact_point = None      # (x, y)
        self.predicted_stump_x = None  # predicted x at the stumps' y-depth
        self.pre_bounce_line = None    # (slope, intercept) x = m*y + b
        self.post_bounce_line = None   # (slope, intercept) x = m*y + b


def _fit_line_x_of_y(points):
    """Fit x = m*y + b using least squares. points: list of (x, y)."""
    pts = np.array(points, dtype=np.float64)
    y = pts[:, 1]
    x = pts[:, 0]
    A = np.vstack([y, np.ones_like(y)]).T
    (m, b), residuals, _, _ = np.linalg.lstsq(A, x, rcond=None)
    if residuals.size > 0:
        err = float(residuals[0])
    else:
        pred = m * y + b
        err = float(np.sum((pred - x) ** 2))
    return m, b, err


def find_pitch_point(points):
    """
    points: chronologically ordered list of (x, y) — y increasing as the
    ball travels from bowler to batsman.
    Returns (pitch_x, pitch_y, pre_line, post_line) or None if there
    aren't enough points to make a confident split.
    """
    n = len(points)
    if n < cfg.MIN_POINTS_FOR_FIT * 2:
        return None

    best_split = None
    best_total_err = float("inf")
    best_pre = None
    best_post = None

    # Leave at least MIN_POINTS_FOR_FIT points on each side of the split
    for split in range(cfg.MIN_POINTS_FOR_FIT, n - cfg.MIN_POINTS_FOR_FIT):
        pre_pts = points[:split]
        post_pts = points[split:]

        m1, b1, err1 = _fit_line_x_of_y(pre_pts)
        m2, b2, err2 = _fit_line_x_of_y(post_pts)
        total_err = err1 + err2

        if total_err < best_total_err:
            best_total_err = total_err
            best_split = split
            best_pre = (m1, b1)
            best_post = (m2, b2)

    if best_split is None:
        return None

    pitch_x, pitch_y = points[best_split]
    return pitch_x, pitch_y, best_pre, best_post


def predict_trajectory(valid_points, stumps_y_depth=cfg.BATSMAN_END_Y):
    """
    Main entry point. valid_points: list of (x, y) confirmed detections
    (chronological order, earliest = closest to bowler).
    """
    result = TrajectoryPrediction()

    if len(valid_points) < cfg.MIN_POINTS_FOR_FIT * 2:
        return result  # not enough data - has_prediction stays False

    split_result = find_pitch_point(valid_points)
    if split_result is None:
        return result

    pitch_x, pitch_y, pre_line, post_line = split_result
    result.pitch_point = (pitch_x, pitch_y)
    result.pre_bounce_line = pre_line
    result.post_bounce_line = post_line

    # Impact point = last confidently tracked point (where the ball
    # reached the pad/bat and tracking typically gets occluded).
    result.impact_point = valid_points[-1]

    # Project the post-bounce line forward to the stumps' depth to get
    # the predicted path the ball *would* have taken.
    m2, b2 = post_line
    predicted_x = m2 * stumps_y_depth + b2
    result.predicted_stump_x = predicted_x
    result.has_prediction = True

    return result
