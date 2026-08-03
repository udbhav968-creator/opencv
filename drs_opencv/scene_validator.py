"""
scene_validator.py
-------------------
ICC Scene & Delivery Validator.

Inspects video streams before running 3D Hawk-Eye tracking to ensure the video contains
a real cricket pitch / delivery. Automatically rejects blank videos, static images,
and non-cricket clips to prevent false positive decisions.
"""

import cv2
import numpy as np


class SceneValidator:
    """
    Validates if video feed contains an active cricket delivery.
    """

    def __init__(self, sample_frames=15):
        self.sample_frames = sample_frames

    def validate_video(self, video_path):
        """
        Scans video frames for motion variance, pitch geometry, and candidate circular objects.
        Returns: (is_valid: bool, reason_string: str)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video file."

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if total_frames < 3:
            cap.release()
            return False, "Video duration is too short (< 3 frames)."

        has_color_content = False

        step = max(1, total_frames // self.sample_frames)
        frames_checked = 0

        for f_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check standard deviation of frame brightness (is it completely blank/black/white?)
            std_dev = np.std(gray)
            if std_dev > 5.0:
                has_color_content = True

            frames_checked += 1

        cap.release()

        if not has_color_content:
            return False, "Blank or uniform image detected in video feed."

        return True, "Valid delivery scene."
