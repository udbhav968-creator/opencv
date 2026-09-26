"""
yolo_detector.py — Deep Learning YOLO Detection Engine for Real DRS
------------------------------------------------------------------
Integrates YOLO ONNX deep learning neural network via OpenCV DNN with seamless
fallback to Ultralytics PyTorch. Features real object detection on COCO class 32 (sports ball).
"""

import cv2
import numpy as np
import logging
import os

try:
    import config as cfg
except ImportError:
    from drs_opencv import config as cfg

logger = logging.getLogger(__name__)

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class YOLODetector:
    """
    YOLO-powered object detector for cricket ball and stump tracking.
    COCO class 32 = 'sports ball'.
    Uses real ONNX neural weights via cv2.dnn for high-performance CPU/GPU execution.
    """

    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.25):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.net = None
        self.is_onnx = False
        self.is_loaded = False

        weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        custom_onnx = os.path.join(weights_dir, 'icc_ball_detector.onnx')
        custom_pt   = os.path.join(weights_dir, 'icc_ball_detector.pt')
        default_pt  = os.path.join(weights_dir, 'yolov8n.pt')

        # Prioritize OpenCV DNN ONNX (universal, zero external dependency, fast C++)
        if os.path.exists(custom_onnx):
            try:
                self.net = cv2.dnn.readNetFromONNX(custom_onnx)
                self.is_onnx = True
                self.is_loaded = True
                logger.info(f"YOLO ONNX neural detector successfully loaded from {custom_onnx}")
            except Exception as e:
                logger.warning(f"Could not load ONNX via cv2.dnn: {e}")

        if not self.is_loaded and YOLO_AVAILABLE:
            load_target = custom_pt if os.path.exists(custom_pt) else (default_pt if os.path.exists(default_pt) else model_name)
            try:
                self.model = YOLO(load_target)
                self.is_loaded = True
                logger.info(f"YOLO PyTorch detector successfully loaded from {load_target}")
            except Exception as exc:
                logger.warning(f"Failed to load PyTorch YOLO weights: {exc}")

    def detect_ball(self, frame_bgr):
        """
        Detects sports ball in frame_bgr using YOLO deep neural network.
        Returns: (x, y, radius, confidence) or None if no ball detected.
        """
        if not self.is_loaded or frame_bgr is None:
            return None

        h_orig, w_orig = frame_bgr.shape[:2]

        # ── 1. ONNX Neural Inference via cv2.dnn ──
        if self.is_onnx and self.net is not None:
            try:
                input_size = 640
                blob = cv2.dnn.blobFromImage(frame_bgr, 1.0 / 255.0, (input_size, input_size), swapRB=True, crop=False)
                self.net.setInput(blob)
                preds = self.net.forward()

                if preds is None or len(preds) == 0:
                    return None

                # Shape: [1, 25200, 85]
                data = preds[0]
                # Filter rows where class 32 (sports ball) score exceeds threshold
                # Indices: 0:4 (cx, cy, w, h), 4: obj_conf, 5+32=37: ball_conf
                obj_conf = data[:, 4]
                ball_conf = data[:, 37]
                scores = obj_conf * ball_conf

                mask = scores > self.confidence_threshold
                if not np.any(mask):
                    return None

                valid_indices = np.where(mask)[0]
                best_idx = valid_indices[np.argmax(scores[valid_indices])]
                best_score = float(scores[best_idx])

                row = data[best_idx]
                cx_norm, cy_norm, w_norm, h_norm = row[0], row[1], row[2], row[3]

                cx = float(cx_norm * (w_orig / input_size))
                cy = float(cy_norm * (h_orig / input_size))
                r = float(max(w_norm, h_norm) * 0.5 * (w_orig / input_size))

                return (cx, cy, max(3.0, r), best_score)

            except Exception as e:
                logger.debug(f"ONNX inference fallback: {e}")
                return None

        # ── 2. Ultralytics PyTorch Inference Fallback ──
        if self.model is not None:
            try:
                results = self.model(frame_bgr, verbose=False, conf=self.confidence_threshold)
                if not results:
                    return None

                best_detection = None
                max_conf = -1.0

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        if cls_id == 32 and conf > max_conf:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            cx = float((x1 + x2) / 2.0)
                            cy = float((y1 + y2) / 2.0)
                            r = float(max(x2 - x1, y2 - y1) / 2.0)
                            max_conf = conf
                            best_detection = (cx, cy, r, conf)

                return best_detection

            except Exception as exc:
                return None

        return None
