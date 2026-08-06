# 🏆 Real DRS Hawk-Eye 3D — Official ICC World Cup Grand Level Suite

> **Official International Standard (ICC World Cup Level) Multi-Page 3D Cricket Decision Review System (DRS).**
> An advanced computer vision and deep neural framework designed for **Grand-Level Live Broadcast Events**:
> - **Page 1: Live DRS Review & RTSP Broadcast Stream Console (`/`)** — Live RTSP IP video stream ingestion, 360° Three.js Floodlight Stadium, UltraEdge Snickometer, AI Voice Commentary, PDF Exporter.
> - **Page 2: Analytics & Model Precision Matrix (`/analytics`)** — 3D Parabolic Height Trajectory Charts, Model Precision-Recall Benchmark Curves, Confusion Matrix, Speed & Spin RPM Distributions.
> - **Page 3: Historical Decision Records Console (`/records`)** — Complete review history database, search & filter controls, CSV export, direct media replay links.
> - **Page 4: Glassmorphic Admin Console (`/admin`)** — Remote training engine controller, live log streaming.

---

## 🌐 Live Application & Production Links

- **Page 1 (Live Review & Stream Ingestion):** [https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)
- **Page 2 (Precision Matrix):** [https://opencv-lyart.vercel.app/analytics](https://opencv-lyart.vercel.app/analytics)
- **Page 3 (Decision Records):** [https://opencv-lyart.vercel.app/records](https://opencv-lyart.vercel.app/records)
- **Page 4 (Admin Console):** [https://opencv-lyart.vercel.app/admin](https://opencv-lyart.vercel.app/admin)
- **API Health Endpoint:** [https://opencv-lyart.vercel.app/health](https://opencv-lyart.vercel.app/health)
- **GitHub Repository:** [https://github.com/udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)

---

## 🏗️ System Architecture & High-Level Flowcharts

### 1️⃣ Grand Multi-Page Application Architecture
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

### 2️⃣ Live RTSP Stream & Broadcast Camera Ingestion Pipeline
```mermaid
flowchart TD
    A[Live Broadcast RTSP IP Stream / HDMI Capture] --> B[OpenCV VideoCapture Ingest Engine]
    B --> C[Real-Time Frame Resolution Normalizer]
    C --> D[Multi-Model Ball Detector Ensemble]
    D --> E[Live Broadcast Hawk-Eye 3D Overlay Renderer]
    E --> F[Multipart MJPEG Stream Response /stream_feed]
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
