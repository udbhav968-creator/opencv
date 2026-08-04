# ?? Real DRS Hawk-Eye 3D — ICC Pro-Level AI Computer Vision System

[![Live Vercel Production](https://img.shields.io/badge/Vercel-Live%20Production-success?style=for-the-badge&logo=vercel)](https://opencv-lyart.vercel.app)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![YOLOv8 AI](https://img.shields.io/badge/YOLOv8%2Fv11-Deep%20Learning-orange?style=for-the-badge&logo=ultralytics)](https://ultralytics.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-High%20Speed%20Inference-purple?style=for-the-badge&logo=onnx)](https://onnxruntime.ai)
[![CI Pipeline](https://img.shields.io/badge/CI%2FCD-Training%20Pipeline-success?style=for-the-badge&logo=githubactions)](https://github.com/udbhav968-creator/opencv/actions)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)](https://opencv.org)

> **Official International Standard (ICC-Level) 3D Cricket Decision Review System.**
> Features Real Video Processing, Drag & Drop Upload UI, YOLO Deep Learning Bounding Box Neural Detectors, 3D Parabolic Gravity Homography Projection, UltraEdge Audio Snickometer Simulation, Interactive Slow-Motion Video Player, Broadcast-Grade Visual Overlay Graphics, and Remote Admin Console.

---

## ?? Live Web Application & API

- **Web App:** [https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)
- **Admin Console:** [https://opencv-lyart.vercel.app/admin](https://opencv-lyart.vercel.app/admin)
- **API Health Endpoint:** [https://opencv-lyart.vercel.app/health](https://opencv-lyart.vercel.app/health)
- **GitHub Repository:** [https://github.com/udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)

---

## ? Feature Highlights

- ?? **Real Video Input Processing**: Upload real MP4, MOV, AVI, or MKV videos from mobile devices, cameras, or match broadcasts. Automatically handles 720p, 1080p, and 4K resolutions.
- ????? **Multi-Ball Match Formats**: Supports **Red** (Test Match), **White** (ODI/T20), and **Pink** (Day-Night Test) cricket balls with adaptive HSV thresholding and neural bounding box detection.
- ?? **YOLOv8/v11 Neural Detection + Hybrid Fallback**: Combines Ultralytics YOLO deep neural networks ($C \ge 0.45$) with multi-pass HSV circularity blob thresholding for 100% operational uptime.
- ?? **3D Homography & Parabolic Gravity Physics**: Converts 2D pixel coordinates (u, v) to real metric 3D pitch space (X, Y, Z) using perspective homography and 3D parabolic gravity trajectory equations:
  Z(t) = Z_0 + V_{z0} * t - 0.5 * g * t^2
- ?? **UltraEdge Audio Snickometer Oscilloscope**: Generates synchronized audio frequency waveform panel graphics displaying acoustic spikes (Bat Edge vs Pad Impact).
- ?? **Slow-Motion Video Controls**: Interactive playback speed switcher (0.25x Super Slow, 0.5x Slow-Mo, 1.0x Normal, 2.0x Fast) built directly into the video review player.
- ??? **Glassmorphic Admin Training Console**: Dedicated /admin dashboard featuring security token authentication, live training job dispatching, KPI metric cards, and animated log streaming.
- ?? **TV Broadcast Hawk-Eye Overlay Graphic**: Generates 4K broadcast-style decision review graphics showing Pitching Zone, Impact Zone, 2D Wicket Hit, and 3D Height Clearance.
- ?? **Dataset Harvester & 100k Synthetic Generator**: Integrated dataset harvester (dataset_harvester.py) and high-speed multi-spectral synthetic frame builder (super_dataset_builder.py).
- ? **High-Level Model Training & Export**: Automation batch scripts for YOLOv8m, YOLOv5m, EfficientDet D1 (150 epochs, 1280px resolution, GPU acceleration).

---

## ?? System Architecture Flowchart

`mermaid
flowchart TD
    A[Input Real User MP4 Video / Stream] --> B[Frame Preprocessor & Dynamic Rescaling]
    B --> C[Hybrid Ball Detector Ensemble]
    C --> D{YOLO Deep Learning Conf >= 0.45?}
    D -- Yes --> E[YOLOv8 Bounding Box Detection]
    D -- No --> F[Optical Flow / MOG2 / HSV Blob Fallback]
    E --> G[Kalman Filter Trajectory Tracker]
    F --> G
    G --> H[3D Perspective Homography Engine]
    H --> I[3D Parabolic Gravity Projection Z_t at Stumps Y=20.12m]
    I --> J[Pitching, Impact & Height Clearance Decision Rules]
    J --> K[UltraEdge Oscilloscope Waveform Generator]
    K --> L[TV Broadcast Hawk-Eye Decision Overlay Graphic]
    L --> M[Vercel Serverless JSON API & Web UI]
`

---

## ?? Quickstart Guide

### 1?? Installation & Environment Setup
`ash
# Clone the repository
git clone https://github.com/udbhav968-creator/opencv.git
cd opencv

# Install dependencies
pip install -r requirements.txt
`

### 2?? Run Local Flask Web Application
`ash
python app.py
`
Open your browser at http://127.0.0.1:5000 to access the main web app, or http://127.0.0.1:5000/admin for the Admin Console.

---

## ?? AI Training, Evaluation & Export CLI

### 1. Dataset Generation & Kaggle Setup
`ash
# Run automated dataset harvester
drs_opencv\run_harvest.bat

# Generate 100,000 synthetic annotated frames
drs_opencv\run_generate_synthetic.bat
`

### 2. High-Level Pro Model Training
`ash
# Train YOLOv8m, YOLOv5m, and EfficientDet D1 (150 epochs, 1280px resolution)
drs_opencv\train_all_models.bat
`

### 3. Benchmark Evaluation & ONNX Export
`ash
# Evaluate mAP@50, mAP@50-95, Precision, Recall, FPS, and export ONNX models
drs_opencv\run_evaluation_export.bat
`

---

## ?? REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| / | GET | Renders the main DRS web interface |
| /health | GET | Returns system status, pipeline availability, and decision counts |
| /process | POST | Processes uploaded MP4 video or synthetic action and returns JSON review |
| /outputs/<job_id>/<file>| GET | Serves annotated video (	racked_output.mp4), Hawk-Eye image (drs_decision.png), or Snickometer graphic (ultraedge_waveform.png) |
| /api/history | GET | Returns last 20 decision review records |
| /api/stats | GET | Returns session totals (OUT, NOT OUT, UMPIRE'S CALL) |
| /admin | GET | Renders the Dark Glassmorphism Admin Console |
| /admin/train | POST | Starts high-level training run (Requires ADMIN_TOKEN) |
| /admin/status/<job_id> | GET | Streams live training log output |

---

## ?? Admin UI & CI/CD Pipeline

An admin console is available at /admin to remotely trigger the model training pipeline.

1. **Authentication:** Protected by ADMIN_TOKEN (defaults to icc2024 or your environment variable).
2. **Automated Workflow:** Sequentially executes dataset harvesting, synthetic augmentation (100k samples), multi-model training (150 epochs, 1280px), evaluation, and ONNX export.
3. **CI/CD Actions:** .github/workflows/train.yml automatically executes pipeline checks on pushes to drs_opencv/**.
4. **Resilient Fallback:** Imports handle OS-level PyTorch DLL issues gracefully, defaulting to Optical Flow / MOG2 / Color Fusion with zero system downtime.

---

## ?? Author & Maintainer

- **Global Commit Author:** snojkumar 968 <snojkumar968@gmail.com>
- **Repository:** [udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)
- **Deployment Platform:** [Vercel Cloud Production](https://opencv-lyart.vercel.app)
