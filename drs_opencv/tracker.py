"""
tracker.py
----------
Tracks the ball across frames. Raw per-frame detections are noisy (motion
blur, partial occlusion by bat/pad, lighting flicker), so we run them
through a Kalman filter to get a smooth trajectory, and we tolerate a
handful of missed detections in a row by coasting on the filter's
prediction.
"""

import cv2
import numpy as np


class BallTracker:
    def __init__(self, max_missed_frames=8):
        self.kalman = cv2.KalmanFilter(4, 2)  # state: x, y, vx, vy | meas: x, y
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        self.initialized = False
        self.missed_frames = 0
        self.max_missed_frames = max_missed_frames

        # Full history of (x, y, radius, frame_index) for confirmed track points
        self.trajectory = []

    def update(self, detection, frame_index):
        """
        detection: (x, y, radius) or None if the ball wasn't found this frame.
        Returns the current best-estimate (x, y) for this frame, or None if
        the track has been lost for too long.
        """
        if detection is not None:
            x, y, radius = detection
            measurement = np.array([[np.float32(x)], [np.float32(y)]])

            if not self.initialized:
                self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
                self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
                self.initialized = True

            self.kalman.predict()
            corrected = self.kalman.correct(measurement)
            est_x, est_y = corrected.flatten()[0].item(), corrected.flatten()[1].item()
            self.missed_frames = 0
            self.trajectory.append((est_x, est_y, radius, frame_index))
            return est_x, est_y

        # No detection this frame -- coast on the filter if we're still
        # within tolerance, otherwise consider the track lost.
        if self.initialized and self.missed_frames < self.max_missed_frames:
            predicted = self.kalman.predict()
            self.missed_frames += 1
            est_x, est_y = predicted.flatten()[0].item(), predicted.flatten()[1].item()
            self.trajectory.append((est_x, est_y, None, frame_index))
            return est_x, est_y

        return None

    def get_trajectory_points(self):
        """Returns list of (x, y) for every tracked frame."""
        return [(p[0], p[1]) for p in self.trajectory]

    def get_valid_trajectory_points(self):
        """
        Returns only points that came from a real detection (radius is not
        None) -- these are the trustworthy ones to fit a curve to.
        """
        return [(p[0], p[1]) for p in self.trajectory if p[2] is not None]
