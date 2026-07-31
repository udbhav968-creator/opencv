"""
ball_detector.py
-----------------
ICC Pro-Level Cricket Ball Detector for Real User-Uploaded Video Footage.

Detects red, white, and pink cricket balls in real-world broadcast & phone videos.
Features adaptive resolution scaling, multi-pass color thresholding, and
motion-blurred contour geometry estimation.
"""

import cv2
import numpy as np
try:
    import config as cfg
except ImportError:
    from drs_opencv import config as cfg


class BallDetector:
    def __init__(self, color_mode="red"):
        """
        color_mode: "red", "white", or "pink" - supports all cricket match formats
        """
        self.color_mode = color_mode.lower()

    def _color_mask(self, hsv_frame):
        if self.color_mode == "white":
            mask = cv2.inRange(
                hsv_frame,
                np.array(cfg.WHITE_BALL_LOWER),
                np.array(cfg.WHITE_BALL_UPPER),
            )
        elif self.color_mode == "pink":
            # Pink ball HSV range (Day-Night Test)
            pink_lower = np.array([140, 50, 100], dtype=np.uint8)
            pink_upper = np.array([175, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_frame, pink_lower, pink_upper)
        else:
            # Red ball HSV ranges (Standard Test)
            mask1 = cv2.inRange(
                hsv_frame,
                np.array(cfg.RED_BALL_LOWER_1),
                np.array(cfg.RED_BALL_UPPER_1),
            )
            mask2 = cv2.inRange(
                hsv_frame,
                np.array(cfg.RED_BALL_LOWER_2),
                np.array(cfg.RED_BALL_UPPER_2),
            )
            mask = cv2.bitwise_or(mask1, mask2)

        # Clean up noise morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        return mask

    def detect(self, frame_bgr):
        """
        Returns (x, y, radius) of ball in real video frame_bgr.
        Supports dynamic resolution scaling (720p, 1080p, 4K).
        """
        if frame_bgr is None:
            return None

        h, w = frame_bgr.shape[:2]

        # Resolution scaling factor relative to 1280x720 baseline
        scale = max(0.5, w / 1280.0)
        min_radius = max(2, int(cfg.MIN_BALL_RADIUS * scale))
        max_radius = max(25, int(cfg.MAX_BALL_RADIUS * scale * 2.5))

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = self._color_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < (3 * scale):
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < min_radius or radius > max_radius:
                continue

            # Circularity metric with support for motion blur elongation
            circle_area = np.pi * (radius ** 2)
            if circle_area == 0:
                continue
            circularity = area / circle_area
            if circularity < 0.40:  # Relaxed for motion-blurred high-speed deliveries
                continue

            score = circularity * area
            if score > best_score:
                best_score = score
                best = (float(x), float(y), float(radius))

        if best is not None:
            return best

        # Second Pass: Hough Circles on Masked Region for fast-moving blurred balls
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        masked_gray = cv2.GaussianBlur(masked_gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(12 * scale),
            param1=50, param2=10,
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is not None:
            c = circles[0][0]
            return float(c[0]), float(c[1]), float(c[2])

        return None
