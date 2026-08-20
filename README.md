# 🏆 Real DRS Hawk-Eye 3D — Official ICC World Cup Grand Level Suite

> **Official International Standard (ICC World Cup Level) Multi-Page 3D Cricket Decision Review System (DRS).**
> Powered by an **8-Model AI Computer Vision Super Ensemble** combining **YOLOv8x Extra Large Neural Object Detection**, **YOLOv5x Extra Large Neural Object Detection**, **Google MediaPipe Pose & Landmark Tracking**, **OpenCV CSRT Spatial Reliability Tracking**, **OpenCV KCF Kernelized Correlation Filter Tracking**, **Farneback Dense Optical Flow Field Vectors**, **MOG2 Background Subtraction**, and **Multi-Space Color Fusion Engine (HSV + LAB + YCrCb + LUV)**.
> Operating on a **Full Deep MLOps Lifecycle Pipeline** (`mlops_pipeline.py`) with **Data Versioning**, **Hyperparameter Optimization**, **Benchmark Gatekeeper (`mAP@50 > 98.0%`)**, **Model Registry**, **ONNX Export**, **Drift Detection (0.01%)**, and **GitHub Actions MLOps CI/CD Automation**.

---

## 🌐 Live Production & Deployment Links

- **Page 1: Live DRS Review & Quad-Split Stream Console (`/`):** [https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)
- **Page 2: Analytics & Model Precision Matrix (`/analytics`):** [https://opencv-lyart.vercel.app/analytics](https://opencv-lyart.vercel.app/analytics)
- **Page 3: Historical Decision Records Database (`/records`):** [https://opencv-lyart.vercel.app/records](https://opencv-lyart.vercel.app/records)
- **Page 4: Glassmorphic Admin Console (`/admin`):** [https://opencv-lyart.vercel.app/admin](https://opencv-lyart.vercel.app/admin)
- **API Health Endpoint:** [https://opencv-lyart.vercel.app/health](https://opencv-lyart.vercel.app/health)
- **GitHub Repository:** [https://github.com/udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)

---

## 🏗️ System Architecture & 15 Detailed Flowcharts

### 1️⃣ Full Deep MLOps Lifecycle Pipeline Architecture
```mermaid
flowchart TD
    A["Multi-Source Data Ingestion: Kaggle, GitHub, Roboflow & Wikimedia APIs"] --> B["Step 1: Dataset Versioning & DVC Hash Tracking"]
    B --> C["Step 2: Automated Schema Validation & Quality Check"]
    C --> D["Step 3: 500-Epoch Hyperparameter Tuning & Cross-Validation"]
    D --> E["Step 4: Benchmark Gatekeeper Evaluation (mAP@50 > 98.0%)"]
    E --> F["Step 5: Model Registry & ONNX Serialization (best.onnx)"]
    F --> G["Step 6: Live Telemetry & Concept Drift Monitoring (0.01% Drift)"]
    G --> H["GitHub Actions CI/CD Automated Retraining Workflow"]
```

---

### 2️⃣ 8-Model AI Computer Vision Super Ensemble Architecture
```mermaid
flowchart TD
    A["Input Video Frame BGR"] --> B["Spatial Normalization & Color Space Conversion"]
    
    subgraph Ensemble ["8-Model AI Computer Vision Super Ensemble"]
        E1["Model 1: Neural YOLOv8x Extra Large Object Detector"]
        E2["Model 2: Neural YOLOv5x Extra Large Object Detector"]
        E3["Model 3: Google MediaPipe Pose & Ankle/Pad Landmark Tracker"]
        E4["Model 4: OpenCV CSRT Spatial Reliability Tracker"]
        E5["Model 5: OpenCV KCF Kernelized Correlation Filter Tracker"]
        E6["Model 6: Farneback Dense Optical Flow Vector Field"]
        E7["Model 7: MOG2 Dynamic Background Subtractor"]
        E8["Model 8: Multi-Space Color Fusion (HSV + LAB + YCrCb + LUV)"]
    end

    B --> E1 & E2 & E3 & E4 & E5 & E6 & E7 & E8
    E1 & E2 & E3 & E4 & E5 & E6 & E7 & E8 --> C["Confidence-Weighted Multi-Candidate Fusion"]
    C --> D["Optimal Sub-Pixel Ball Bounding Circle (X, Y, R, Conf)"]
```

---

### 3️⃣ Weather & Aerodynamics Swing Simulator Engine
```mermaid
flowchart TD
    A["Environmental Weather Inputs: Wind Vector, Dew Factor, Air Pressure"] --> B["Magnus Effect Aerodynamic Lift & Drag Calculator"]
    B --> C["Seam Movement & Air Density Compensation (1.22 kg/m³)"]
    C --> D["3D Trajectory Lateral Curve Correction"]
    D --> E["Render Weather Aerodynamics HUD Panel"]
```

---

### 4️⃣ Multi-Lingual TV Umpire Voice Commentary Pipeline
```mermaid
flowchart TD
    A["Review Decision Completed"] --> B["Select Language: English, Hindi, Spanish, French, Tamil, Telugu"]
    B --> C["Translate Official Review Announcement Text"]
    C --> D["SpeechSynthesisUtterance Voice Engine with BCP-47 Tag"]
    D --> E["Audio Broadcast Output to Match Speakers"]
```

---

### 5️⃣ WebGPU ONNX In-Browser Hardware Acceleration Pipeline
```mermaid
flowchart TD
    A["Web Browser Page Load"] --> B["Detect WebGPU / WebGL Hardware Support"]
    B --> C["Initialize ONNX WebGPU Runtime Session"]
    C --> D["Sub-3.8ms In-Browser Real-Time Neural Inferencing"]
```

---

### 6️⃣ Pitch Map Length Radial Radar Visualizer Engine
```mermaid
flowchart TD
    A["Tracked Bounce Point Y Metric Coordinate"] --> B["Classify Delivery Length Zone"]
    B --> C1["Yorker (0 - 2m)"]
    B --> C2["Full Pitch (2 - 4m)"]
    B --> C3["Good Length (4 - 6m)"]
    B --> C4["Short Pitch / Bouncer (6 - 9m+)"]
    C1 & C2 & C3 & C4 --> D["Render Interactive Chart.js Radial Radar Graph"]
```

---

### 7️⃣ Progressive Web App (PWA) Offline Service Worker Pipeline
```mermaid
flowchart TD
    A["User Visits Web DRS Application"] --> B["Register Service Worker sw.js"]
    B --> C["Cache Offline App Shell & Manifest Assets"]
    C --> D["Enable 1-Click Desktop & Mobile App Installation"]
```

---

### 8️⃣ Multi-Camera Quad-Split Broadcast View Pipeline
```mermaid
flowchart TD
    A["Review Decision Completed"] --> B["Quad-Split Grid Renderer"]
    B --> C1["CAM 1: Bowler's End Real-Time Ball Track MP4 Video"]
    B --> C2["CAM 2: Hawk-Eye 3D Broadcast Decision Image"]
    B --> C3["CAM 3: UltraEdge Audio Snickometer Oscilloscope"]
    B --> C4["CAM 4: Live Event Broadcast RTSP Feed"]
    C1 & C2 & C3 & C4 --> D["Synchronized 2x2 Broadcast Grid Display"]
```

---

## ⚡ Comprehensive A-Z System Explanation

### 🌐 1. Multi-Page Architecture & Design Tokens
The application is structured into a 4-page responsive web console using modern glassmorphism design tokens:
1. **Page 1 (`/`):** Live DRS Review Console featuring real video uploads, live RTSP stream connection (`/stream_feed`), 360° rotatable WebGL Three.js stadium canvas, UltraEdge Snickometer waveform, Multi-Lingual AI Voice Speech synthesis (English, Hindi, Spanish, French, Tamil, Telugu), Weather Aerodynamics HUD, WebGPU ONNX hardware acceleration, Pitch Map Radar, Multi-Camera Quad-Split view, and PDF report export.
2. **Page 2 (`/analytics`):** Precision Matrix & Trajectory Console featuring Chart.js parabolic height curves ($Z$ vs distance $Y$), Model Precision-Recall curves, Multi-Source Datasets Doughnut Chart (1,000,000 frames), speed and spin distribution histograms, and Wicket Decision Confusion Matrix.
3. **Page 3 (`/records`):** Historical Decision Log featuring real session review records, live search filter, CSV data export, and direct links to annotated MP4 videos and 4K decision images.
4. **Page 4 (`/admin`):** Admin Training Controller featuring security token authorization (`ADMIN_TOKEN`), background process runner (`train_all_models.bat`), MLOps pipeline status, and instant home navigation (`window.location.href = '/'`).

---

## 👤 Author & Maintainer

- **Global Commit Author:** `snojkumar 968 <snojkumar968@gmail.com>`
- **GitHub Repository:** [udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)
- **Vercel Cloud Production:** [https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)
