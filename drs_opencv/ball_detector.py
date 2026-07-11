"""
ball_detector.py
-----------------
Detects the cricket ball in a single video frame using HSV colour
thresholding + contour geometry (circularity check). This keeps the demo
dependency-light (no trained model needed) while still being robust to
lighting noise, since we filter by both colour AND "how circular" the
blob is.
"""

import cv2
import numpy as np
import config as cfg


class BallDetector:
    def __init__(self, color_mode="red"):
        """
        color_mode: "red" or "white" -- pick based on the ball used in
        your footage. Red is the default (Test-match ball).
        """
        self.color_mode = color_mode

    def _color_mask(self, hsv_frame):
        if self.color_mode == "white":
            mask = cv2.inRange(
                hsv_frame,
                np.array(cfg.WHITE_BALL_LOWER),
                np.array(cfg.WHITE_BALL_UPPER),
            )
        else:
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

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        return mask

    def detect(self, frame_bgr):
        """
        Returns (x, y, radius) of the most likely ball location in this
        frame, or None if nothing convincing was found.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = self._color_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 4:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < cfg.MIN_BALL_RADIUS or radius > cfg.MAX_BALL_RADIUS:
                continue

            # Circularity: how close is this contour's area to a perfect
            # circle of the same radius? 1.0 = perfect circle.
            circle_area = np.pi * (radius ** 2)
            if circle_area == 0:
                continue
            circularity = area / circle_area
            if circularity < 0.55:
                continue  # too irregular to be the ball

            # Prefer larger, more-circular blobs
            score = circularity * area
            if score > best_score:
                best_score = score
                best = (float(x), float(y), float(radius))

        if best is not None:
            return best

        # Fallback: sometimes the ball's contour merges with another
        # similarly-coloured region (e.g. crossing a red-ish background
        # element), tanking circularity. In that case, run Hough circle
        # detection restricted to the masked region as a second pass.
        masked_gray = cv2.bitwise_and(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), mask=mask
        )
        masked_gray = cv2.GaussianBlur(masked_gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
            param1=60, param2=12,
            minRadius=cfg.MIN_BALL_RADIUS, maxRadius=cfg.MAX_BALL_RADIUS,
        )
        if circles is not None:
            c = circles[0][0]  # take strongest response
            return float(c[0]), float(c[1]), float(c[2])

        return None
