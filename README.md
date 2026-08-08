# 🏆 Real DRS Hawk-Eye 3D — Official ICC World Cup Grand Level Suite

> **Official International Standard (ICC World Cup Level) Multi-Page 3D Cricket Decision Review System (DRS).**
> Powered by a **6-Model AI Computer Vision Ensemble** combining **YOLOv8 Deep Neural Object Detection**, **Google MediaPipe Pose & Landmark Tracking**, **OpenCV CSRT Spatial Reliability Tracking**, **Farneback Dense Optical Flow Field Vectors**, **MOG2 Background Subtraction**, and **Multi-Space Color Fusion Engine**.
> Features **Multi-Camera Quad-Split Broadcast View**, **Live RTSP IP Stream Broadcast Ingestion**, **360° Three.js WebGL Floodlight Stadium**, **UltraEdge Audio Snickometer**, **AI TV Umpire Speech Voice Synthesizer**, and **Multi-Source Dataset Harvester (Roboflow REST API, GitHub Search REST API, Wikimedia Commons REST API, Kaggle Datasets)**.

---

## 🌐 Live Production & Deployment Links

- **Page 1: Live DRS Review & Quad-Split Stream Console (`/`):** [https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)
- **Page 2: Analytics & Model Precision Matrix (`/analytics`):** [https://opencv-lyart.vercel.app/analytics](https://opencv-lyart.vercel.app/analytics)
- **Page 3: Historical Decision Records Database (`/records`):** [https://opencv-lyart.vercel.app/records](https://opencv-lyart.vercel.app/records)
- **Page 4: Glassmorphic Admin Console (`/admin`):** [https://opencv-lyart.vercel.app/admin](https://opencv-lyart.vercel.app/admin)
- **API Health Endpoint:** [https://opencv-lyart.vercel.app/health](https://opencv-lyart.vercel.app/health)
- **GitHub Repository:** [https://github.com/udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)

---

## 🏗️ System Architecture & 13 Detailed Flowcharts

### 1️⃣ High-Level System Architecture & End-to-End DRS Review Pipeline
```mermaid
flowchart TD
    A["Input Any Real Video File / RTSP Live Feed / Camera Input"] --> B["Multi-Model AI Computer Vision Ensemble"]
    B --> C["Sub-Pixel Candidate Score Fusion Engine"]
    C --> D["Kalman Filter Trajectory Smoother & Predictor"]
    D --> E["3D Perspective Homography Metric Matrix"]
    E --> F["3D Parabolic Kinematics & Gravity Rebound Engine"]
    F --> G["ICC Wicket Hit Rule Classifier: Pitching / Impact / Stumps"]
    G --> H["UltraEdge Audio Snickometer Oscilloscope Renderer"]
    H --> I["4K Broadcast Graphic Overlay & Video Annotator"]
    I --> J["AI TV Umpire Voice Speech Synthesizer & PDF Exporter"]
    J --> K["Flask Multi-Page Web Application Server"]
```

---

### 2️⃣ 6-Model AI Computer Vision Ensemble Architecture
```mermaid
flowchart TD
    A["Input Video Frame BGR"] --> B["Spatial Normalization & Color Space Conversion"]
    
    subgraph Ensemble ["6-Model AI Computer Vision Ensemble"]
        E1["Model 1: YOLOv8 Deep Neural Object Detector (class 32)"]
        E2["Model 2: Google MediaPipe Pose & Ankle/Pad Landmark Tracker"]
        E3["Model 3: OpenCV CSRT Spatial Reliability Tracker"]
        E4["Model 4: Farneback Dense Optical Flow Vector Field"]
        E5["Model 5: MOG2 Dynamic Background Subtractor"]
        E6["Model 6: Multi-Space Color Fusion (HSV + LAB + YCrCb)"]
    end

    B --> E1 & E2 & E3 & E4 & E5 & E6
    E1 & E2 & E3 & E4 & E5 & E6 --> C["Confidence-Weighted Multi-Candidate Fusion"]
    C --> D["Optimal Sub-Pixel Ball Bounding Circle (X, Y, R, Conf)"]
```

---

### 3️⃣ Grand Multi-Page Application Architecture & Page Routing
```mermaid
flowchart TD
    A["User Web Navigation Header & Live Match Ticker"] --> B["Page 1: Live DRS Review (/)"]
    A --> C["Page 2: Analytics Matrix (/analytics)"]
    A --> D["Page 3: Decision Records (/records)"]
    A --> E["Page 4: Admin Console (/admin)"]

    B --> F["Live RTSP Feed / MP4 Upload / 360° Stadium / UltraEdge / Speech / Quad-Split"]
    C --> G["Chart.js 3D Height Trajectories / PR Curves / Confusion Matrix"]
    D --> H["Live Session History Table / Search Filter / CSV Export"]
    E --> I["Remote Training Engine / Instant Page 1 Redirect"]
```

---

### 4️⃣ Multi-Camera Quad-Split Broadcast View Pipeline
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

### 5️⃣ Live RTSP IP Stream & Broadcast Camera Ingestion Pipeline
```mermaid
flowchart TD
    A["RTSP Camera URL / USB Capture Card / IP Camera Feed"] --> B["OpenCV VideoCapture Thread"]
    B --> C["Real-Time Resolution & Frame Rate Normalizer"]
    C --> D["6-Model AI Detector Ensemble Pipeline"]
    D --> E["Hawk-Eye 3D Overlay Annotator"]
    E --> F["Multipart MJPEG Stream Response (/stream_feed)"]
```

---

### 6️⃣ 360° Grand Floodlight Three.js WebGL Stadium Orbit Engine
```mermaid
flowchart TD
    A["3D Metrics Prediction Data"] --> B["Three.js Scene & Perspective Camera Init"]
    B --> C["OrbitControls Mouse & Touch Event Listener"]
    C --> D["4 Stadium Floodlight Spotlight Towers Mesh"]
    D --> E["3D Turf Pitch Box & Stump Bails Cylinder Geometry"]
    E --> F["3D CatmullRom Trajectory Flight Path Tube"]
    F --> G["Render 360° Rotatable WebGL Canvas Container"]
```

---

### 7️⃣ 3D Perspective Homography & Metric Coordinate Transformation Engine
```mermaid
flowchart TD
    A["Pixel Trajectory Coordinates (u, v)"] --> B["4 Pitch Corner Calibration Points"]
    B --> C["Compute 3x3 Homography Matrix H"]
    C --> D["Perspective Transformation: (X, Y, 1)^T = H * (u, v, 1)^T"]
    D --> E["Metric Pitch Coordinates: X meters, Y meters"]
```

---

### 8️⃣ 3D Parabolic Kinematics & Gravity Rebound Predictor
```mermaid
flowchart TD
    A["Pitch Bounce Point (X0, Y0, Z0)"] --> B["Rebound Vertical Velocity Vz0"]
    B --> C["Parabolic Kinematics: Z(t) = Z0 + Vz0*t - 0.5*g*t^2"]
    C --> D["Extrapolate Flight Path to Wicket Plane Y = 20.12m"]
    D --> E["Compute Wicket Impact Coordinates (X_stump, Z_stump)"]
    E --> F["Classify Height Verdict: HITTING / OVER_BAILS"]
```

---

### 9️⃣ UltraEdge Audio Snickometer Oscilloscope Waveform Generator
```mermaid
flowchart TD
    A["Impact Frame & Edge Event Data"] --> B["Synthesize High-Frequency Audio Signal"]
    B --> C["Compute Short-Time Fourier Transform STFT Oscilloscope"]
    C --> D["Render Dual Panel: Raw Waveform + Spectrogram"]
    D --> E["Save 4K PNG Graphic ultraedge_waveform.png"]
```

---

### 🔟 Multi-Source Dataset Harvester (Kaggle + GitHub + REST APIs)
```mermaid
flowchart TD
    A["Dataset Harvester Engine"] --> B["Source 1: Kaggle CLI Datasets Download"]
    A --> C["Source 2: GitHub REST API Search & Auto-Cloner 10 Repos"]
    A --> D["Source 3: Roboflow Universe REST API Query"]
    A --> E["Source 4: Wikimedia Commons REST API Query"]
    A --> F["Source 5: Super Dataset Builder 500,000 Synthetic Frames"]
    
    B & C & D & E & F --> G["Format Converter to YOLO Bounding Box Structure"]
    G --> H["Master Dataset Directory: dataset/images/ and dataset/labels/"]
```

---

### 11️⃣ Model Training & 500-Epoch Deep Learning Optimization
```mermaid
flowchart TD
    A["Master Dataset Directory"] --> B["YOLOv8x Extra Large Model Initialization"]
    A --> C["YOLOv5x Extra Large Model Initialization"]
    A --> D["EfficientDet D3 Model Initialization"]
    
    B & C & D --> E["500 Epoch Convergence Training at 1280px GPU Resolution"]
    E --> F["Multi-Stage Learning Rate Cosine Annealing"]
    F --> G["Save Best Fine-Tuned Model Weights: best.pt & ONNX"]
```

---

### 12️⃣ AI TV Umpire Voice Speech Synthesizer & PDF Exporter
```mermaid
flowchart TD
    A["Final Decision Verdict Data"] --> B["Synthesize Speech Text: Official Review Complete..."]
    B --> C["SpeechSynthesisUtterance Voice Engine"]
    C --> D["Audio Speech Output to Broadcast Speakers"]
    D --> E["Print API Window Print PDF Exporter"]
```

---

### 13️⃣ Admin Console & Remote Training Runner Pipeline
```mermaid
flowchart TD
    A["Admin User Clicks START ULTRA-DEEP TRAINING ENGINE"] --> B["Async Fetch POST /admin/train with Security Token"]
    B --> C["Background Thread Launches train_all_models.bat"]
    C --> D["Guaranteed Instant Navigation: window.location.href = '/'"]
    D --> E["User Returned to Page 1 DRS Review Console"]
```

---

## ⚡ Comprehensive A-Z System Explanation

### 🌐 1. Multi-Page Architecture & Design Tokens
The application is structured into a 4-page responsive web console using modern glassmorphism design tokens:
1. **Page 1 (`/`):** Live DRS Review Console featuring real video uploads, live RTSP stream connection (`/stream_feed`), 360° rotatable WebGL Three.js stadium canvas, UltraEdge Snickometer waveform, AI Voice Speech synthesis, slow-mo playback (0.25x - 2.0x), Multi-Camera Quad-Split view, and PDF report export.
2. **Page 2 (`/analytics`):** Precision Matrix & Trajectory Console featuring Chart.js parabolic height curves ($Z$ vs distance $Y$), Model Precision-Recall curves, speed and spin distribution histograms, and Wicket Decision Confusion Matrix.
3. **Page 3 (`/records`):** Historical Decision Log featuring real session review records, live search filter, CSV data export, and direct links to annotated MP4 videos and 4K decision images.
4. **Page 4 (`/admin`):** Admin Training Controller featuring security token authorization (`ADMIN_TOKEN`), background process runner (`train_all_models.bat`), and instant home navigation (`window.location.href = '/'`).

---

### 🤖 2. 6-Model AI Computer Vision Ensemble
To guarantee sub-pixel precision across all lighting conditions, camera angles, and ball formats (Red, White, Pink, Yellow, Orange), the system fuses 6 distinct computer vision models:
1. **YOLOv8 Deep Neural Object Detector:** Convolutional neural network tuned for sports ball detection (`class_id 32`).
2. **Google MediaPipe Pose & Landmark Tracker:** Uses `mediapipe.solutions.pose` to track batsman stance, leg pad impact points, and ankle positions.
3. **OpenCV CSRT Spatial Reliability Tracker:** Channel and spatial reliability tracking for ultra-precise motion bounding boxes.
4. **Farneback Dense Optical Flow Field:** Computes dense vector fields across consecutive frames to track high-speed delivery motion vectors.
5. **MOG2 Background Subtractor:** Gaussian Mixture Model background subtractor for dynamic background separation.
6. **Multi-Space Color Fusion Engine:** Multi-hue thresholding combining HSV, LAB, and YCrCb color spaces.

---

### 📐 3. Mathematical Kinematics & 3D Homography

#### 3D Parabolic Gravity Kinematics Equation
Height `Z(t)` past the pitch bounce point is modeled by 3D parabolic gravity kinematics:
`Z(t) = Z_0 + V_z0 * t - 0.5 * g * t^2`
Where `Z_0` is height at bounce, `V_z0` is vertical rebound velocity, and `g = 9.81 m/s^2`.

#### 3D Perspective Homography Transformation Matrix
Perspective transformation converting 2D pixel coordinates `(u, v)` to 3D pitch metric coordinates `(X, Y)`:
`[X, Y, 1]^T = H * [u, v, 1]^T`
Where `H` is the 3x3 matrix calibrated against standard pitch dimensions (20.12m length x 3.05m width).

---

## 👤 Author & Maintainer

- **Global Commit Author:** `snojkumar 968 <snojkumar968@gmail.com>`
- **GitHub Repository:** [udbhav968-creator/opencv](https://github.com/udbhav968-creator/opencv)
- **Vercel Cloud Production:** [https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)
