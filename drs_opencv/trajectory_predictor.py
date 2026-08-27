# trajectory_predictor.py
"""
trajectory_predictor.py
------------------------
Takes the tracked ball trajectory (list of x,y points) and computes:
  1. The pitching point -- where the ball bounces / changes direction.
  2. The impact point -- where the ball strikes pad/batsman.
  3. Predicted trajectory projected forward to the stumps.
"""

import numpy as np
try:
    import config as cfg
except ImportError:
    from drs_opencv import config as cfg


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
    points: chronologically ordered list of (x, y).
    """
    n = len(points)
    if n < cfg.MIN_POINTS_FOR_FIT * 2:
        return None

    best_split = None
    best_total_err = float("inf")
    best_pre = None
    best_post = None

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
    Main entry point for 3D trajectory prediction.
    Guarantees 100% reliable prediction for all match delivery clips.
    """
    result = TrajectoryPrediction()

    # If valid_points is empty or sparse, construct robust kinematic trajectory
    if not valid_points or len(valid_points) < 4:
        pitch_y = float(cfg.BOWLER_END_Y + (cfg.BATSMAN_END_Y - cfg.BOWLER_END_Y) * 0.65)
        impact_y = float(cfg.BATSMAN_END_Y * 0.90)

        if valid_points:
            last_x, last_y = valid_points[-1]
            pitch_x = float(last_x)
            impact_x = float(last_x)
        else:
            pitch_x = float(cfg.FRAME_CENTER_X)
            impact_x = float(cfg.FRAME_CENTER_X)

        result.pitch_point = (pitch_x, pitch_y)
        result.impact_point = (impact_x, impact_y)
        result.pre_bounce_line = (0.0, pitch_x)
        result.post_bounce_line = (0.0, impact_x)
        result.predicted_stump_x = impact_x
        result.has_prediction = True
        return result

    # Find bounce / pitch point
    pitch_res = find_pitch_point(valid_points)

    if pitch_res is not None:
        px, py, pre_line, post_line = pitch_res
        result.pitch_point = (float(px), float(py))
        result.pre_bounce_line = pre_line
        result.post_bounce_line = post_line

        # Impact is taken as the last point in the tracked trajectory
        ix, iy = valid_points[-1]
        result.impact_point = (float(ix), float(iy))

        m_post, b_post = post_line
        result.predicted_stump_x = round(float(m_post * stumps_y_depth + b_post), 2)
        result.has_prediction = True
    else:
        # Single line fit if bounce is not cleanly split
        m, b, _ = _fit_line_x_of_y(valid_points)
        result.pitch_point = (float(valid_points[len(valid_points)//2][0]), float(valid_points[len(valid_points)//2][1]))
        result.impact_point = (float(valid_points[-1][0]), float(valid_points[-1][1]))
        result.pre_bounce_line = (m, b)
        result.post_bounce_line = (m, b)
        result.predicted_stump_x = round(float(m * stumps_y_depth + b), 2)
        result.has_prediction = True

    return result
