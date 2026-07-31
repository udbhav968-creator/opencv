"""
stats_analyzer.py
-----------------
Computes delivery analytics from the tracked trajectory:
  - Estimated ball speed (px/frame -> km/h equivalent)
  - Swing/seam movement (lateral deviation pre-pitch)
  - Trajectory straightness index
  - Pre-pitch vs post-pitch angle change (seam movement after bounce)
"""

import math
import numpy as np


class DeliveryStatsAnalyzer:
    """
    Analyse the smoothed ball trajectory to compute delivery statistics.
    All measurements are in pixel space; speed is in px/frame unless
    a pixels-per-metre calibration is supplied.
    """

    def __init__(self, fps=25.0, px_per_metre=None):
        self.fps           = fps
        self.px_per_metre  = px_per_metre

    def analyze(self, trajectory_points):
        """
        Compute delivery stats from a list of trajectory points.
        Supports 2-tuples (x, y), 3-tuples (frame, x, y), or (x, y, r).
        """
        if not trajectory_points or len(trajectory_points) < 4:
            return self._empty()

        parsed = []
        for pt in trajectory_points:
            if len(pt) == 2:
                parsed.append((float(pt[0]), float(pt[1])))
            elif len(pt) == 3:
                # Check if first element is frame index (< 500)
                if pt[0] < 500 and pt[1] > 0:
                    parsed.append((float(pt[1]), float(pt[2])))
                else:
                    parsed.append((float(pt[0]), float(pt[1])))
            elif len(pt) > 3:
                parsed.append((float(pt[1]), float(pt[2])))

        if len(parsed) < 4:
            return self._empty()

        pts = np.array(parsed, dtype=float)

        speed        = self._estimate_speed(pts)
        swing        = self._estimate_swing(pts)
        angle_change = self._estimate_angle_change(pts)
        straightness = self._straightness_index(pts)
        description  = self._describe(speed, swing, angle_change)

        return {
            "speed":        round(float(speed), 2),
            "speed_unit":   "km/h" if self.px_per_metre else "px/frame",
            "swing":        round(float(swing), 1),
            "angle_change": round(float(angle_change), 1),
            "straightness": round(float(straightness), 3),
            "description":  description,
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _estimate_speed(self, pts):
        """Mean inter-frame displacement, optionally converted to km/h."""
        diffs   = np.diff(pts, axis=0)
        dists   = np.hypot(diffs[:, 0], diffs[:, 1])
        mean_px = float(np.mean(dists)) if dists.size > 0 else 0.0
        if self.px_per_metre:
            m_per_frame  = mean_px / self.px_per_metre
            m_per_second = m_per_frame * self.fps
            return m_per_second * 3.6  # km/h
        return mean_px

    def _estimate_swing(self, pts):
        """Lateral (X-axis) deviation across the whole trajectory."""
        xs = pts[:, 0]
        return float(xs.max() - xs.min()) if xs.size > 0 else 0.0

    def _estimate_angle_change(self, pts):
        """Angle between first-half direction and second-half direction."""
        mid = len(pts) // 2
        if mid < 2:
            return 0.0
        v1  = pts[mid - 1] - pts[0]
        v2  = pts[-1]      - pts[mid]
        a1  = math.atan2(v1[1], v1[0])
        a2  = math.atan2(v2[1], v2[0])
        deg = math.degrees(abs(a2 - a1))
        return min(deg, 360.0 - deg)

    def _straightness_index(self, pts):
        """Ratio of straight-line distance to total path length."""
        straight = float(np.linalg.norm(pts[-1] - pts[0]))
        path_len = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
        if path_len < 1e-6:
            return 1.0
        return min(straight / path_len, 1.0)

    def _describe(self, speed, swing, angle_change):
        parts = []
        if self.px_per_metre:
            if speed > 130:
                parts.append("Express pace delivery")
            elif speed > 110:
                parts.append("Fast-medium delivery")
            elif speed > 90:
                parts.append("Medium-pace delivery")
            else:
                parts.append("Slow delivery")
        if swing > 30:
            parts.append("significant lateral movement")
        elif swing > 15:
            parts.append("moderate swing/seam")
        if angle_change > 10:
            parts.append(f"sharp angle change after pitch ({angle_change:.0f}°)")
        return (", ".join(parts) + ".") if parts else "Straight delivery with minimal movement."

    @staticmethod
    def _empty():
        return {
            "speed":        0.0,
            "speed_unit":   "px/frame",
            "swing":        0.0,
            "angle_change": 0.0,
            "straightness": 0.0,
            "description":  "Insufficient data to compute stats.",
        }
