"""
confidence_scorer.py
--------------------
Scores the reliability of each ball detection based on multiple signals:
  - contour circularity
  - radius plausibility (vs config MIN/MAX)
  - inter-frame position consistency
  - colour mask coverage strength

Used by the tracker to weight Kalman filter measurements.
"""

import math
import numpy as np
import config as cfg


class DetectionConfidenceScorer:
    """
    Assigns a confidence score [0.0 – 1.0] to each raw detection.
    Higher-confidence detections have more influence on the Kalman state.
    """

    def __init__(self):
        self._prev_position = None
        self._frame_index   = 0

    def score(self, detection, frame=None):
        """
        Score a detection tuple (cx, cy, radius) or None.

        Args:
            detection: (cx, cy, radius) from BallDetector, or None
            frame:     Current BGR frame (optional, used for colour check)

        Returns:
            float in [0.0, 1.0]
        """
        if detection is None:
            self._prev_position = None
            self._frame_index  += 1
            return 0.0

        cx, cy, radius = detection
        scores = []

        # 1. Radius plausibility
        scores.append(self._radius_score(radius))

        # 2. Inter-frame velocity plausibility
        scores.append(self._velocity_score(cx, cy))

        # 3. Colour mask strength (if frame provided)
        if frame is not None:
            scores.append(self._colour_score(frame, cx, cy, radius))

        confidence = float(np.mean(scores))
        self._prev_position = (cx, cy)
        self._frame_index  += 1
        return round(min(max(confidence, 0.0), 1.0), 3)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _radius_score(self, radius):
        """Score how well the detected radius fits the configured ball size."""
        min_r = cfg.MIN_BALL_RADIUS
        max_r = cfg.MAX_BALL_RADIUS
        if min_r <= radius <= max_r:
            # Highest score at the midpoint of expected range
            mid   = (min_r + max_r) / 2
            span  = (max_r - min_r) / 2
            return 1.0 - abs(radius - mid) / (span + 1e-6) * 0.3
        # Out-of-range penalty
        over  = max(0, radius - max_r)
        under = max(0, min_r - radius)
        penalty = (over + under) / (max_r - min_r + 1e-6)
        return max(0.0, 1.0 - penalty)

    def _velocity_score(self, cx, cy):
        """Penalise detections that imply an implausibly large jump from the previous frame."""
        if self._prev_position is None:
            return 0.85  # first detection — moderate prior
        px, py = self._prev_position
        dist   = math.hypot(cx - px, cy - py)
        # Expect ball to move at most ~80 px/frame at typical frame rate
        max_expected = 80.0
        score  = max(0.0, 1.0 - dist / (max_expected * 3))
        return score

    def _colour_score(self, frame, cx, cy, radius):
        """
        Sample pixel colour at the detection centre and check how well it
        matches the expected ball hue. Returns 0.5–1.0.
        """
        try:
            import cv2
            x, y = int(cx), int(cy)
            h, w = frame.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                return 0.5
            patch_r = max(1, int(radius * 0.4))
            x0, x1 = max(0, x - patch_r), min(w, x + patch_r)
            y0, y1 = max(0, y - patch_r), min(h, y + patch_r)
            patch  = frame[y0:y1, x0:x1]
            if patch.size == 0:
                return 0.5
            hsv    = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mean_s = float(hsv[:, :, 1].mean())   # saturation
            mean_v = float(hsv[:, :, 2].mean())   # brightness
            # Ball should be saturated and bright
            score  = (mean_s / 255.0) * 0.5 + (mean_v / 255.0) * 0.5
            return round(0.5 + score * 0.5, 3)    # scale to [0.5, 1.0]
        except Exception:
            return 0.5
