# 🏆 Real DRS Hawk-Eye 3D — Official ICC World Cup Grand Level Suite

> **Official International Standard (ICC World Cup Level) Multi-Page 3D Cricket Decision Review System (DRS).**
> Powered by a **6-Model AI Computer Vision Ensemble** combining **YOLOv8 Deep Learning**, **Google MediaPipe Pose & Landmark Tracking**, **OpenCV CSRT Spatial Reliability Tracking**, **Farneback Dense Optical Flow**, **MOG2 Background Subtraction**, and **Multi-Space Color Fusion**.

---

## 🌐 Live Application & Production Links

- **Page 1 (Live Review & RTSP Stream Ingestion):** [https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)
- **Page 2 (Precision Matrix):** [https://opencv-lyart.vercel.app/analytics](https://opencv-lyart.vercel.app/analytics)
- **Page 3 (Decision Records):** [https://opencv-lyart.vercel.app/records](https://opencv-lyart.vercel.app/records)
- **Page 4 (Admin Console):** [https://opencv-lyart.vercel.app/admin](https://opencv-lyart.vercel.app/admin)
- **API Health Endpoint:** [https://opencv-lyart.vercel.app/health](https://opencv-lyart.vercel.app/health)
- **GitHub Repository:** [https://github.com/udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)

---

## 🏗️ System Architecture & High-Level Flowcharts

### 1️⃣ 6-Model AI Computer Vision Ensemble Architecture
```mermaid
flowchart TD
    A[Input Any Real Cricket Video / MP4 / RTSP Stream] --> B[Frame Preprocessor & Spatial Rescaling]
    B --> C[Multi-Model AI Computer Vision Ensemble]
    
    subgraph Ensemble ["6-Model AI Computer Vision Ensemble"]
        E1[YOLOv8 Deep Neural Object Detector]
        E2[Google MediaPipe Pose & Landmark Tracker]
        E3[OpenCV CSRT Spatial Reliability Tracker]
        E4[Farneback Dense Optical Flow Vector Field]
        E5[MOG2 Dynamic Background Subtractor]
        E6[Multi-Space HSV/LAB Color Fusion Engine]
    end

    C --> E1 & E2 & E3 & E4 & E5 & E6
    E1 & E2 & E3 & E4 & E5 & E6 --> F[Multi-Candidate Score Fusion Engine]
    F --> G[Kalman Filter Trajectory Tracker]
    G --> H[3D Homography Metric Conversion]
    H --> I[3D Parabolic Gravity Predictor]
    I --> J[ICC DRS Decision Rules Engine & Hit Probability %]
    J --> K[UltraEdge Oscilloscope Audio Simulator]
    K --> L[4K Broadcast Hawk-Eye Overlay Generator]
    L --> M[AI TV Voice Speech Synthesizer & PDF Exporter]
    M --> N[Vercel Serverless API & Web UI]
```

### 2️⃣ Grand Multi-Page Application Architecture
```mermaid
flowchart TD
    A[User Web Navigation Header & Live Match Ticker] --> B[Page 1: Live DRS Review & RTSP Ingestion /]
    A --> C[Page 2: Analytics & Precision Matrix /analytics]
    A --> D[Page 3: Decision Records Log /records]
    A --> E[Page 4: Admin Training Console /admin]

    B --> F[Video Upload / RTSP Stream Ingest / 360° Stadium / UltraEdge / AI Voice]
    C --> G[Trajectory Charts / PR Curves / Confusion Matrix]
    D --> H[Search History / CSV Export / Replay Links]
    E --> I[Authenticated Pro Training Engine Runner]
```

### 3️⃣ 360° Grand Floodlight Three.js WebGL Stadium Orbit Engine
```mermaid
flowchart TD
    A[3D Metrics Generated] --> B[Three.js Scene & Perspective Camera Init]
    B --> C[OrbitControls Touch & Mouse Handler]
    C --> D[4 Stadium Floodlight Tower Spotlights Mesh]
    D --> E[3D Turf Pitch & Stump Geometry Cylinder Mesh]
    E --> F[3D CatmullRom Trajectory Flight Path Tube]
    F --> G[Render 360° Rotatable WebGL Canvas Container]
```

---

## ⚡ Mathematical Foundations & Physics Engine

### 1. 3D Parabolic Trajectory Equations
Ball height `Z(t)` at any time step `t` past the pitch bounce point is governed by 3D parabolic gravity kinematics:
`Z(t) = Z_0 + V_z0 * t - 0.5 * g * t^2`
Where:
- `Z_0` = Height of ball at pitch bounce point (m)
- `V_z0` = Vertical rebound velocity (m/s)
- `g` = Acceleration due to gravity (9.81 m/s²)

### 2. Perspective 3D Homography Mapping
Converts image pixel coordinates `(u, v)` to metric pitch coordinates `(X, Y)`:
`[X, Y, 1]^T = H * [u, v, 1]^T`
Where `H` is the `3x3` perspective transformation matrix calibrated against standard pitch dimensions (20.12m x 3.05m).

---

## 👤 Author & Maintainer

- **Global Commit Author:** `snojkumar 968 <snojkumar968@gmail.com>`
- **Repository:** [udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)
- **Deployment Platform:** [Vercel Cloud Production](https://opencv-lyart.vercel.app)
