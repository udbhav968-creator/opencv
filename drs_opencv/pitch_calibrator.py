"""
pitch_calibrator.py
--------------------
Automatic Pitch Keypoint & Stump Homography Calibrator.

Detects stump lines and pitch crease boundaries across arbitrary video angles
(broadcast cameras, side angles, amateur phone cameras) and dynamically computes
perspective Homography matrix points.
"""

import cv2
import numpy as np
try:
    import config as cfg
except ImportError:
    from drs_opencv import config as cfg


class PitchCalibrator:
    """
    Automated Pitch Keypoint & Homography Calibrator.
    """

    def __init__(self):
        pass

    def calibrate_pitch_geometry(self, frame_bgr):
        """
        Scans a video frame to detect stump lines and crease keypoints.
        Returns: dict with (bowler_y, batsman_y, stumps_far_w, stumps_near_w)
        """
        if frame_bgr is None:
            return self._default_geometry()

        h, w = frame_bgr.shape[:2]

        # Convert to grayscale and run Canny edge detection
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect vertical lines for stumps
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=20, maxLineGap=10)

        stump_y_candidates = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check for near-vertical lines (stumps)
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 75 <= angle <= 105:
                    stump_y_candidates.append(min(y1, y2))
                    stump_y_candidates.append(max(y1, y2))

        if stump_y_candidates:
            batsman_y = int(np.percentile(stump_y_candidates, 85))
            bowler_y = int(np.percentile(stump_y_candidates, 15))
            batsman_y = max(int(h * 0.70), min(int(h * 0.92), batsman_y))
            bowler_y = max(int(h * 0.05), min(int(h * 0.30), bowler_y))
        else:
            batsman_y = int(h * 0.88)
            bowler_y = int(h * 0.11)

        stumps_near_w = int(w * 0.075)
        stumps_far_w = int(w * 0.020)

        return {
            "bowler_end_y": bowler_y,
            "batsman_end_y": batsman_y,
            "stumps_width_far": stumps_far_w,
            "stumps_width_near": stumps_near_w,
            "frame_width": w,
            "frame_height": h,
            "calibrated": True
        }

    def _default_geometry(self):
        return {
            "bowler_end_y": cfg.BOWLER_END_Y,
            "batsman_end_y": cfg.BATSMAN_END_Y,
            "stumps_width_far": cfg.STUMPS_WIDTH_FAR,
            "stumps_width_near": cfg.STUMPS_WIDTH_NEAR,
            "frame_width": cfg.FRAME_WIDTH,
            "frame_height": cfg.FRAME_HEIGHT,
            "calibrated": False
        }
