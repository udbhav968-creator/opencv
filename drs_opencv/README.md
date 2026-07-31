# 📂 drs_opencv — Core Engine Modules

This directory contains the Python modules powering the **Real DRS Hawk-Eye 3D Computer Vision System**.

## 🛠 Core Module Architecture

| Module | Description |
| :--- | :--- |
| `main.py` | Primary pipeline orchestrator connecting video input, tracking, 3D physics, and graphics export. |
| `dataset_manager.py` | Kaggle API dataset downloader & synthetic YOLO annotation generator (`dataset.yaml`). |
| `train_yolo.py` | PyTorch training pipeline using Ultralytics YOLOv8 / YOLOv11 for cricket ball detection. |
| `evaluate_model.py` | Benchmark suite for calculating mAP@50, mAP@50-95, Precision, Recall, and FPS. |
| `export_model.py` | Exporter converting PyTorch `.pt` weights to ONNX (`.onnx`) & TorchScript. |
| `yolo_detector.py` | Hybrid Ball Detector combining YOLO neural inference ($C \ge 0.45$) with HSV fallback. |
| `ball_detector.py` | Adaptive HSV color blob detector supporting Red, White, and Pink cricket balls. |
| `tracker.py` | 4-state Kalman Filter tracking ball position $(x, y, v_x, v_y)$ across motion blur. |
| `drs_3d_engine.py` | Perspective Homography engine mapping 2D pixels $(u, v)$ to 3D pitch space $(X, Y, Z)$. |
| `physics_3d_predictor.py` | 3D Parabolic Gravity Trajectory Predictor calculating height clearance at stumps ($Y = 20.12\text{m}$). |
| `ultraedge.py` | UltraEdge Audio Snickometer simulation generating high-frequency waveforms. |
| `hawk_eye_visualizer.py` | TV broadcast graphics renderer producing 4K decision review images. |
| `ai_verdict.py` | Generates human-readable AI verdict explanations and confidence scores. |
| `stats_analyzer.py` | Computes delivery speed (km/h), swing/seam deviation, and straightness index. |

---

## ⚡ Execution Commands

```bash
# Run full pipeline on sample video
python drs_opencv/main.py --input sample_input.mp4 --output_dir output_test

# Generate synthetic dataset
python drs_opencv/dataset_manager.py --generate-synthetic

# Train YOLO detector
python drs_opencv/train_yolo.py --epochs 10

# Benchmark model performance
python drs_opencv/evaluate_model.py

# Export ONNX model
python drs_opencv/export_model.py --format onnx
```
