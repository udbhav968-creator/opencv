"""
frame_preprocessor.py
---------------------
Pre-processing pipeline applied to each video frame before ball detection.
Improves detection accuracy under challenging lighting or motion conditions.
"""

import cv2
import numpy as np


class FramePreprocessor:
    """
    Apply a configurable chain of image-processing operations to each frame
    to improve ball detection robustness.

    Default pipeline:
      1. Gaussian blur        — reduces sensor noise
      2. CLAHE on L-channel   — adaptive contrast enhancement
      3. Optional sharpening  — enhances ball edges

    All operations are applied in-place-safe (returns a new array).
    """

    def __init__(self, blur_ksize=3, clahe_clip=2.0, clahe_grid=8, sharpen=False):
        """
        Args:
            blur_ksize:  Gaussian kernel size (odd integer, 0 to disable)
            clahe_clip:  CLAHE clip limit (0.0 to disable CLAHE)
            clahe_grid:  CLAHE tile grid size
            sharpen:     Apply unsharp mask after CLAHE
        """
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.sharpen    = sharpen

        if clahe_clip > 0:
            self._clahe = cv2.createCLAHE(
                clipLimit    = clahe_clip,
                tileGridSize = (clahe_grid, clahe_grid)
            )
        else:
            self._clahe = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Pre-process a single BGR frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            Pre-processed BGR image
        """
        out = frame.copy()

        # 1. Gaussian denoise
        if self.blur_ksize > 1:
            out = cv2.GaussianBlur(out, (self.blur_ksize, self.blur_ksize), 0)

        # 2. CLAHE adaptive contrast on L channel (LAB colour space)
        if self._clahe is not None:
            lab       = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b   = cv2.split(lab)
            l_clahe   = self._clahe.apply(l)
            lab       = cv2.merge([l_clahe, a, b])
            out       = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. Optional unsharp mask (sharpening)
        if self.sharpen:
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
            out     = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)

        return out

    def process_batch(self, frames):
        """Pre-process a list of frames and return a new list."""
        return [self.process(f) for f in frames]

    @staticmethod
    def resize_for_detection(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Resize a frame to the detection target resolution using INTER_AREA
        (best quality for downscaling).
        """
        h, w = frame.shape[:2]
        if (w, h) == (target_w, target_h):
            return frame
        interp = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LINEAR
        return cv2.resize(frame, (target_w, target_h), interpolation=interp)
