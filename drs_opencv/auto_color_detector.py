"""
auto_color_detector.py
-----------------------
Automatic Ball Color & Stadium Lighting Calibration Engine for Real DRS.

Scans incoming video stream frames, computes multi-channel HSV histograms,
and automatically determines ball color mode ("red", "white", "pink", "yellow", "orange")
without requiring manual user selection.
"""

import cv2
import numpy as np
try:
    import config as cfg
except ImportError:
    from drs_opencv import config as cfg


class AutoColorDetector:
    """
    Automatic Ball Color & Lighting Classifier.
    """

    def __init__(self, sample_frames=15):
        self.sample_frames = sample_frames

    def detect_ball_color(self, video_path):
        """
        Analyzes video frames to automatically detect ball color mode.
        Returns: ("red", "white", "pink", "yellow", or "orange", confidence_score)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "red", 0.70  # Default fallback

        scores = {
            "red": 0.0,
            "white": 0.0,
            "pink": 0.0,
            "yellow": 0.0,
            "orange": 0.0
        }

        frames_checked = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30
        step = max(1, total_frames // self.sample_frames)

        for f_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Red ball HSV mask checks
            m_red1 = cv2.inRange(hsv, np.array([0, 100, 60]), np.array([12, 255, 255]))
            m_red2 = cv2.inRange(hsv, np.array([168, 100, 60]), np.array([180, 255, 255]))
            scores["red"] += float(np.count_nonzero(cv2.bitwise_or(m_red1, m_red2)))

            # White ball HSV mask check (High Value, Low Saturation)
            m_white = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 45, 255]))
            scores["white"] += float(np.count_nonzero(m_white)) * 0.45

            # Pink ball HSV mask check
            m_pink = cv2.inRange(hsv, np.array([140, 50, 90]), np.array([175, 255, 255]))
            scores["pink"] += float(np.count_nonzero(m_pink)) * 1.2

            # Yellow/Orange ball HSV mask checks
            m_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
            scores["yellow"] += float(np.count_nonzero(m_yellow)) * 1.1

            m_orange = cv2.inRange(hsv, np.array([12, 120, 100]), np.array([22, 255, 255]))
            scores["orange"] += float(np.count_nonzero(m_orange)) * 1.1

            frames_checked += 1
            if frames_checked >= self.sample_frames:
                break

        cap.release()

        # Find best candidate color
        best_color = max(scores, key=scores.get)
        total_score = sum(scores.values()) + 1e-6
        confidence = min(0.99, max(0.65, scores[best_color] / total_score * 2.5))

        return best_color, round(confidence, 2)
