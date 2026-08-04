# ?? Real DRS Hawk-Eye 3D — ICC Pro-Level AI Computer Vision System

> **Official International Standard (ICC-Level) 3D Cricket Decision Review System (DRS).**
> An advanced computer vision and deep learning framework featuring real video input processing, 4-model ensemble ball detection, 3D parabolic gravity trajectory projection, UltraEdge audio snickometer oscilloscope simulation, interactive slow-motion playback, broadcast visual overlay graphics, and a remote glassmorphism admin console.

---

## ?? Live Application Links

- **Main Web Application:** [https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)
- **Glassmorphism Admin Console:** [https://opencv-lyart.vercel.app/admin](https://opencv-lyart.vercel.app/admin)
- **API Health Endpoint:** [https://opencv-lyart.vercel.app/health](https://opencv-lyart.vercel.app/health)
- **GitHub Repository:** [https://github.com/udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)

---

## ??? System Architecture & High-Level Flowcharts

### 1?? End-to-End System Processing Pipeline
`mermaid
flowchart TD
    A[Input Real User MP4 / Broadcast Stream] --> B[Frame Preprocessor & Spatial Rescaling]
    B --> C[Scene & Motion Validator]
    C --> D[Multi-Model Ball Detector Ensemble]
    
    subgraph Ensemble ["4-Model Computer Vision Ensemble"]
        E1[YOLOv8 Deep Learning Detector]
        E2[Farneback Dense Optical Flow Engine]
        E3[MOG2 Background Subtractor]
        E4[Multi-Space HSV/LAB Color Fusion]
    end

    D --> E1 & E2 & E3 & E4
    E1 & E2 & E3 & E4 --> F[Multi-Candidate Score Fusion Engine]
    F --> G[Kalman Filter Trajectory Tracker]
    G --> H[3D Homography Metric Conversion]
    H --> I[3D Parabolic Gravity Predictor]
    I --> J[ICC DRS Decision Rules Engine]
    J --> K[UltraEdge Oscilloscope Audio Simulator]
    K --> L[4K Broadcast Hawk-Eye Overlay Generator]
    L --> M[Vercel Serverless API & Web UI]
`

### 2?? 3D Metric Coordinate Transformation Flowchart
`mermaid
flowchart LR
    A["2D Frame Pixel (u, v)"] --> B["Camera Intrinsic Matrix K"]
    B --> C["Perspective Homography Matrix H"]
    C --> D["2D Ground Pitch Point (X, Y)"]
    D --> E["3D Parabolic Ball Trajectory Z(t)"]
    E --> F["Impact Zone Y=17.5m"]
    E --> G["Stump Target Plane Y=20.12m"]
    F --> H["Lateral & Height Clearance Verdict"]
    G --> H
`

### 3?? High-Level Pro Training & CI/CD Pipeline
`mermaid
flowchart TD
    A["Trigger Run (GitHub Push / Admin UI)"] --> B["Dataset Harvester (Kaggle & GitHub)"]
    B --> C["Super Dataset Builder (100k Synthetic Frames)"]
    C --> D["YOLOv8 Medium Training (150 Epochs, 1280px)"]
    C --> E["YOLOv5 Medium Training (150 Epochs, 1280px)"]
    C --> F["EfficientDet D1 Training (150 Epochs, 1280px)"]
    D & E & F --> G["Model Benchmark & Evaluation Suite"]
    G --> H["High-Speed ONNX Exporter"]
    H --> I["Deploy Artifacts to GitHub / Vercel"]
`

### 4?? UltraEdge Audio Snickometer Acoustic Spectrum Analysis
`mermaid
flowchart TD
    A[Acoustic Audio Signal Stream] --> B[High-Pass Frequency Filter]
    B --> C[Spectral FFT Energy Density Analysis]
    C --> D{Peak Frequency >= 2500Hz?}
    D -- Yes --> E[Bat Edge Spike Detected - WOOD CONTACT]
    D -- No --> F[Pad Impact Pulse Detected - CLOTH/PAD]
    E --> G[Render Oscilloscope Waveform Graphic Panel]
    F --> G
`

---

## ? Mathematical Foundations & Physics Engine

### 1. 3D Parabolic Trajectory Equations
Ball height Z(t) at any time step 	 past the pitch bounce point is governed by 3D parabolic gravity kinematics:
Z(t) = Z_0 + V_z0 * t - 0.5 * g * t^2
Where:
- Z_0 = Height of ball at pitch bounce point (m)
- V_z0 = Vertical rebound velocity (m/s)
- g = Acceleration due to gravity (9.81 m/s²)

### 2. Perspective 3D Homography Mapping
Converts image pixel coordinates (u, v) to metric pitch coordinates (X, Y):
[X, Y, 1]^T = H * [u, v, 1]^T
Where H is the 3x3 perspective transformation matrix calibrated against standard pitch dimensions (20.12m x 3.05m).

---

## ? Comprehensive Feature Matrix

- ?? **Real Video Input Processing**: Real-time parsing of MP4, MOV, AVI, and MKV files across 720p, 1080p, and 4K resolutions.
- ????? **Universal Ball Color Calibration**: Supports Red (Test Matches), White (ODIs/T20s), Pink (Day-Night Tests), and Yellow (Practice) balls with auto-detection.
- ?? **4-Model Computer Vision Ensemble**:
  1. **YOLOv8/v11 Deep Neural Detector**: Bounding box object detection (C >= 0.45).
  2. **Farneback Dense Optical Flow**: Motion-field vector tracking across consecutive frames.
  3. **MOG2 Background Subtractor**: Dynamic background separation.
  4. **Multi-Space Color Fusion**: HSV + LAB + YCrCb color space thresholding.
- ?? **UltraEdge Audio Snickometer**: Renders a synchronized 2D oscilloscope audio waveform visualizer.
- ?? **Slow-Motion Video Controls**: Interactive playback rate switcher (0.25x Super Slow, 0.5x Slow-Mo, 1.0x Normal, 2.0x Fast).
- ??? **Glassmorphism Admin Console**: /admin dashboard featuring security token authentication, live training job dispatching, KPI metric cards, and animated log streaming.
- ?? **TV Broadcast Hawk-Eye Overlay**: Generates 4K broadcast-style decision review graphics showing Pitching Zone, Impact Zone, 2D Wicket Hit, and 3D Height Clearance.

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

### 1. Dataset Harvesting & Synthetic Generation
`ash
# Run automated Kaggle & GitHub harvester
drs_opencv\run_harvest.bat

# Generate 100,000 multi-spectral synthetic frames
drs_opencv\run_generate_synthetic.bat
`

### 2. High-Level Pro Model Training
`ash
# Train YOLOv8m, YOLOv5m, and EfficientDet D1 (150 epochs, 1280px resolution, GPU)
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

## ?? Author & Maintainer

- **Global Commit Author:** snojkumar 968 <snojkumar968@gmail.com>
- **Repository:** [udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)
- **Deployment Platform:** [Vercel Cloud Production](https://opencv-lyart.vercel.app)
