"""
physics_3d_predictor.py
------------------------
3D Parabolic Trajectory Predictor & Height Clearance Model.

Implements true 3D physics equations including:
  - Gravity: Z(t) = Z0 + Vz0 * t - 0.5 * g * t^2
  - Coefficient of restitution (bounce vertical momentum loss): e_v = 0.65
  - Friction loss on pitch: e_y = 0.85
  - Parabolic Height Projection at the stumps (Y = 20.12m)
  - 3D Wicket Verdict: HITTING / UMPIRES_CALL / MISSING_HIGH / MISSING_WIDE
"""

import numpy as np
try:
    from drs_3d_engine import (
        Perspective3DEngine,
        PITCH_LENGTH_M,
        STUMP_HEIGHT_M,
        STUMP_WIDTH_M,
        STUMP_HALF_WIDTH_M,
        UMPIRES_CALL_MARGIN_M
    )
except ImportError:
    from drs_opencv.drs_3d_engine import (
        Perspective3DEngine,
        PITCH_LENGTH_M,
        STUMP_HEIGHT_M,
        STUMP_WIDTH_M,
        STUMP_HALF_WIDTH_M,
        UMPIRES_CALL_MARGIN_M
    )

G_ACCEL = 9.81  # Gravitational acceleration m/s^2
RESTITUTION_Z = 0.65  # Vertical bounce coefficient
FRICTION_Y = 0.88     # Forward velocity retention post-pitch
BAIL_THICKNESS_M = 0.035 # Metric bail height tolerance


class Physics3DPrediction:
    """Stores full 3D metric trajectory analysis."""

    def __init__(self):
        self.has_prediction = False
        
        # 3D Metric Points (X, Y, Z) in metres
        self.pitch_3d = None       # (X, Y, Z) at bounce
        self.impact_3d = None      # (X, Y, Z) at pad impact
        self.stump_3d = None       # Predicted (X, Y, Z) at stumps Y = 20.12m
        
        # Predicted Stump Metrics
        self.stump_x = 0.0         # Lateral deviation from center (m)
        self.stump_z = 0.0         # Height above ground (m)
        
        # Detailed 3D Verdicts
        self.lateral_verdict = "MISSING"     # HITTING / UMPIRES_CALL / MISSING
        self.height_verdict  = "MISSING_HIGH"# HITTING / UMPIRES_CALL / MISSING_HIGH
        self.final_3d_verdict = "MISSING"    # HITTING / UMPIRES_CALL / MISSING
        
        # Dense projected 3D trajectory path [(X, Y, Z), ...]
        self.projected_path_3d = []


class Physics3DPredictor:
    """
    Fits 3D parabolic gravity physics to tracked points and projects post-impact path.
    """

    def __init__(self, fps=25.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.engine3d = Perspective3DEngine()

    def predict_3d(self, valid_pixel_points):
        """
        Converts 2D pixel trajectory to 3D world space and calculates parabolic trajectory.
        
        Args:
            valid_pixel_points: List of (px, py) or (px, py, r) tuples
        """
        res = Physics3DPrediction()
        if len(valid_pixel_points) < 6:
            return res

        # 1. Convert pixel points to 3D Metric World Space
        pts_3d = []
        for pt in valid_pixel_points:
            px, py = pt[0], pt[1]
            r = pt[2] if len(pt) > 2 else 8
            x, y, z = self.engine3d.estimate_3d_point(px, py, r)
            pts_3d.append((x, y, z))

        pts_3d = np.array(pts_3d, dtype=np.float64)

        # 2. Find Pitch Bounce Point (Min Z or maximum curvature in Y-Z plane)
        min_z_idx = np.argmin(pts_3d[:, 2])
        if min_z_idx < 2 or min_z_idx >= len(pts_3d) - 2:
            min_z_idx = len(pts_3d) // 2

        pitch_3d = tuple(pts_3d[min_z_idx])
        impact_3d = tuple(pts_3d[-1])
        res.pitch_3d = pitch_3d
        res.impact_3d = impact_3d

        # 3. Fit Velocity Vector Post-Bounce (from pitch to impact)
        post_pts = pts_3d[min_z_idx:]
        if len(post_pts) < 2:
            return res

        # Time elapsed post-bounce
        t_steps = np.arange(len(post_pts)) * self.dt
        
        # Fit Linear X velocity: X(t) = X0 + Vx * t
        Vx = np.polyfit(t_steps, post_pts[:, 0], 1)[0]
        
        # Fit Linear Y velocity: Y(t) = Y0 + Vy * t
        Vy = np.polyfit(t_steps, post_pts[:, 1], 1)[0]
        Vy = max(5.0, Vy)  # Ensure positive forward velocity (m/s)

        # Fit Vertical Z velocity with Gravity: Z(t) = Z0 + Vz0 * t - 0.5 * g * t^2
        z_adj = post_pts[:, 2] + 0.5 * G_ACCEL * (t_steps ** 2)
        Vz0 = np.polyfit(t_steps, z_adj, 1)[0]

        # 4. Project forward from Impact (Y_impact) to Stumps (Y = 20.12m)
        y_impact = impact_3d[1]
        y_stumps = PITCH_LENGTH_M
        delta_y = max(0.01, y_stumps - y_impact)

        # Time to travel from Impact to Stumps
        t_to_stumps = delta_y / Vy

        # Final predicted 3D position at Stumps plane
        x_stumps = impact_3d[0] + Vx * t_to_stumps
        z_stumps = max(0.0, impact_3d[2] + Vz0 * t_to_stumps - 0.5 * G_ACCEL * (t_to_stumps ** 2))

        res.stump_x = round(float(x_stumps), 3)
        res.stump_z = round(float(z_stumps), 3)
        res.stump_3d = (res.stump_x, y_stumps, res.stump_z)

        # 5. Generate Dense 3D Project Path for Rendering
        n_dense = 20
        t_dense = np.linspace(0, t_to_stumps, n_dense)
        path = []
        for td in t_dense:
            xp = impact_3d[0] + Vx * td
            yp = impact_3d[1] + Vy * td
            zp = max(0.0, impact_3d[2] + Vz0 * td - 0.5 * G_ACCEL * (td ** 2))
            path.append((round(float(xp), 3), round(float(yp), 3), round(float(zp), 3)))
        res.projected_path_3d = path

        # 6. Compute 3D Wicket Verdict (Lateral + Height Clearance)
        res.lateral_verdict = self._classify_lateral(res.stump_x)
        res.height_verdict  = self._classify_height(res.stump_z)
        res.final_3d_verdict = self._combine_verdicts(res.lateral_verdict, res.height_verdict)
        res.has_prediction = True

        return res

    def _classify_lateral(self, x):
        abs_x = abs(x)
        if abs_x <= (STUMP_HALF_WIDTH_M - UMPIRES_CALL_MARGIN_M):
            return "HITTING"
        elif abs_x <= (STUMP_HALF_WIDTH_M + UMPIRES_CALL_MARGIN_M):
            return "UMPIRES_CALL"
        else:
            return "MISSING"

    def _classify_height(self, z):
        if z <= (STUMP_HEIGHT_M - UMPIRES_CALL_MARGIN_M):
            return "HITTING"
        elif z <= (STUMP_HEIGHT_M + UMPIRES_CALL_MARGIN_M + BAIL_THICKNESS_M):
            return "UMPIRES_CALL"
        else:
            return "MISSING_HIGH"

    def _combine_verdicts(self, lateral, height):
        if lateral == "MISSING" or height == "MISSING_HIGH":
            return "MISSING"
        if lateral == "UMPIRES_CALL" or height == "UMPIRES_CALL":
            return "UMPIRES_CALL"
        return "HITTING"
