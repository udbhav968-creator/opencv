"""
evaluate_model.py
-----------------
ICC Pro-Level Model Evaluation & Benchmark Suite for DRS Ball Detector.

Calculates key performance metrics:
- Precision, Recall, mAP@50, mAP@50-95
- Average Inference Latency (ms/frame)
- Frame Processing Rate (FPS)
"""

import os
import sys
import time
import argparse
import logging
import json
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError as err:
    YOLO = None
    logger.warning(f"Ultralytics module not imported: {err}")

from dataset_manager import YAML_PATH, generate_synthetic_icc_dataset

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

def evaluate_icc_model(weights_path=None, data_yaml=YAML_PATH):
    """Evaluates detector model and benchmarks inference speed."""
    if weights_path is None:
        weights_path = os.path.join(WEIGHTS_DIR, "icc_ball_detector.pt")
        if not os.path.exists(weights_path):
            weights_path = "yolov8n.pt"

    logger.info(f"Starting DRS Model Benchmark (Weights: '{weights_path}', Data Config: '{data_yaml}')...")

    if YOLO is None:
        raise RuntimeError("Ultralytics package is required to run real YOLO validation benchmarks. Please install via 'pip install ultralytics'.")

    model = YOLO(weights_path)

    if not os.path.exists(data_yaml):
        data_yaml = generate_synthetic_icc_dataset(num_train=30, num_val=10)

    val_results = model.val(data=data_yaml, verbose=False)

    precision = float(val_results.results_dict.get('metrics/precision(B)', 0.0))
    recall    = float(val_results.results_dict.get('metrics/recall(B)', 0.0))
    map50     = float(val_results.results_dict.get('metrics/mAP50(B)', 0.0))
    map50_95  = float(val_results.results_dict.get('metrics/mAP50-95(B)', 0.0))

    # Benchmark Inference Latency
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(dummy_img, verbose=False)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model.predict(dummy_img, verbose=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(times))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    logger.info(f"Precision: {precision:.3f} | Recall: {recall:.3f} | mAP@50: {map50:.3f} | mAP@50-95: {map50_95:.3f} | Latency: {avg_latency:.2f}ms/frame | FPS: {fps:.1f}")

    return {
        'precision': precision,
        'recall': recall,
        'map50': map50,
        'map50_95': map50_95,
        'latency_ms': avg_latency,
        'fps': fps
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DRS Model Evaluation Suite")
    parser.add_argument('--weights', type=str, default=None, help="Path to custom model weights")
    args = parser.parse_args()

    results = evaluate_icc_model(args.weights)
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved to 'evaluation_results.json'.")

