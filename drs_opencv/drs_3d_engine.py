"""
drs_3d_engine.py
----------------
3D Camera Projection and World Coordinate Transformation Engine.

Converts 2D image coordinates (u, v) into real-world 3D cricket pitch coordinates (X, Y, Z)
using homography matrices and camera perspective projections.

World Coordinate System:
  - X axis: Lateral position across the pitch in metres (0.0 = middle stump, +ve = off, -ve = leg)
  - Y axis: Depth along pitch in metres (0.0 = bowler's crease, 20.12m = batsman's stumps)
  - Z axis: Vertical height above ground in metres (0.0 = pitch surface, 0.711m = top of bails)
"""

import numpy as np
try:
    import config as cfg
except ImportError:
    from drs_opencv import config as cfg

# Real-world Cricket Standard Dimensions (in metres)
PITCH_LENGTH_M = 20.12     # 22 yards
STUMP_HEIGHT_M = 0.7112    # 28 inches
STUMP_WIDTH_M  = 0.2286    # 9 inches
STUMP_HALF_WIDTH_M = STUMP_WIDTH_M / 2.0
BAIL_THICKNESS_M = 0.03    # ~3cm bail thickness

# Umpire's Call margins in real metres (50% of ball diameter = ~3.6cm)
BALL_RADIUS_M = 0.036
UMPIRES_CALL_MARGIN_M = BALL_RADIUS_M


class Perspective3DEngine:
    """
    Handles perspective homography projection between 2D pixel space and 3D metric world space.
    """

    def __init__(self, frame_w: int = cfg.FRAME_WIDTH, frame_h: int = cfg.FRAME_HEIGHT):
        self.frame_w = frame_w
        self.frame_h = frame_h
        
        # 4 Corner Points in 2D Pixel space (Bowler Far-Left, Far-Right, Batsman Near-Left, Near-Right)
        self.src_pts = np.float32([
            [cfg.FRAME_CENTER_X - cfg.STUMPS_WIDTH_FAR * 1.5, cfg.BOWLER_END_Y],
            [cfg.FRAME_CENTER_X + cfg.STUMPS_WIDTH_FAR * 1.5, cfg.BOWLER_END_Y],
            [cfg.FRAME_CENTER_X - cfg.STUMPS_WIDTH_NEAR * 1.5, cfg.BATSMAN_END_Y],
            [cfg.FRAME_CENTER_X + cfg.STUMPS_WIDTH_NEAR * 1.5, cfg.BATSMAN_END_Y]
        ])

        # Corresponding 4 Corner Points in 3D Metric World Space (X, Y)
        self.dst_pts = np.float32([
            [-1.5, 0.0],
            [ 1.5, 0.0],
            [-1.5, PITCH_LENGTH_M],
            [ 1.5, PITCH_LENGTH_M]
        ])

        # Compute Homography Matrix H (Pixel -> Metric Ground Plane)
        self.H_pixel_to_world = self._compute_homography(self.src_pts, self.dst_pts)
        self.H_world_to_pixel = np.linalg.inv(self.H_pixel_to_world)

    def _compute_homography(self, src, dst):
        """Computes 3x3 Homography transformation matrix."""
        A = []
        for i in range(4):
            u, v = src[i]
            x, y = dst[i]
            A.append([-u, -v, -1, 0, 0, 0, u*x, v*x, x])
            A.append([0, 0, 0, -u, -v, -1, u*y, v*y, y])
        A = np.array(A, dtype=np.float64)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        return H / H[2, 2]

    def pixel_to_world_2d(self, px: float, py: float):
        """Maps a 2D pixel coordinate (px, py) to ground 3D world coordinate (X, Y)."""
        pt = np.array([px, py, 1.0], dtype=np.float64)
        res = self.H_pixel_to_world @ pt
        res /= res[2]
        return float(res[0]), float(res[1])

    def world_to_pixel_2d(self, X: float, Y: float):
        """Maps ground 3D world coordinate (X, Y) back to 2D pixel coordinate (px, py)."""
        pt = np.array([X, Y, 1.0], dtype=np.float64)
        res = self.H_world_to_pixel @ pt
        res /= res[2]
        return float(res[0]), float(res[1])

    def estimate_3d_point(self, px: float, py: float, radius_px: float):
        """Estimates full (X, Y, Z) world point from 2D pixel coordinates and ball radius."""
        X, Y = self.pixel_to_world_2d(px, py)

        t = Y / PITCH_LENGTH_M
        t = max(0.0, min(1.0, t))
        expected_ground_r = cfg.MIN_BALL_RADIUS + t * (cfg.MAX_BALL_RADIUS - cfg.MIN_BALL_RADIUS)

        m_per_pixel = 3.0 / (cfg.STUMPS_WIDTH_FAR + t * (cfg.STUMPS_WIDTH_NEAR - cfg.STUMPS_WIDTH_FAR))

        delta_r = radius_px - expected_ground_r if radius_px else 0.0
        Z = max(0.0, delta_r * m_per_pixel * 2.5) if radius_px else 0.2

        return round(float(X), 3), round(float(Y), 3), round(float(Z), 3)
