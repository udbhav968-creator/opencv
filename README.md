# 🏏 Real DRS — Hawk-Eye 3D Cricket Decision Review System

> A high-level, production-grade Computer Vision + 3D Physics + AI web application that simulates the official cricket Decision Review System (DRS) — **3D Homography → Parabolic Trajectory Physics → UltraEdge Snickometer → TV Broadcast Graphic Rendering**.

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?style=flat-square&logo=opencv)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=flat-square&logo=vercel)](https://opencv-lyart.vercel.app)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 🌐 Live Demo

**[https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)**

---

## ✨ Features (Real DRS Level)

- 📐 **3D Homography & Metric Projection (`drs_3d_engine.py`)** — Converts 2D pixel coordinates $(u, v)$ to real-world 3D metric coordinates $(X, Y, Z)$ using standard pitch dimensions ($20.12\text{m}$ pitch length, $0.711\text{m}$ stump height).
- 🚀 **3D Parabolic Gravity Physics (`physics_3d_predictor.py`)** — Fits vertical parabolic equations $Z(t) = Z_0 + V_{z0} t - \frac{1}{2} g t^2$ under gravity ($9.81\text{ m/s}^2$) to calculate exact ball height clearance over bails at the stumps plane.
- 🎙️ **UltraEdge / Snickometer Oscilloscope (`ultraedge.py`)** — Simulates audio frequency contact signatures (Bat Edge: $2500\text{Hz}-4000\text{Hz}$ sharp spike vs Pad Impact: $300\text{Hz}-600\text{Hz}$ dull pulse).
- 📺 **Hawk-Eye Broadcast Multi-View Renderer (`hawk_eye_visualizer.py`)** — Generates a 4-panel $1200\times 800$ TV Broadcast card featuring:
  1. Main Track View with metric callouts
  2. 3D Side Height View (showing parabolic height clearance over $0.711\text{m}$ stumps)
  3. 3D Pitch Top-Down Map View (showing lateral trajectory deviation)
  4. UltraEdge Snickometer Oscilloscope Panel
- 🤖 **AI Verdict Explainer (`ai_verdict.py`)** — Natural language analysis explaining the LBW law decision, confidence scoring, and tips.
- 📊 **Delivery Stats Analyzer (`stats_analyzer.py`)** — Computes speed (km/h), swing/seam deviation, angle changes post-bounce, and trajectory straightness index.
- 🧹 **Frame Preprocessor (`frame_preprocessor.py`)** — CLAHE adaptive contrast in LAB space + Gaussian denoise + unsharp mask sharpening.
- 🌐 **Flask Web App & Vercel Serverless Deployment** — Interactive dark-mode dashboard with instant synthetic generation and custom video processing.

---

## 🏗️ Project Structure

```
opencv/
├── app.py                        # Flask web application (main entry point)
├── requirements.txt              # Python dependencies (optimized for Vercel <250MB)
├── vercel.json                   # Vercel deployment configuration
├── .python-version               # Pinned Python 3.12 for Vercel
├── templates/
│   └── index.html                # Dark-mode Hawk-Eye 3D web UI
└── drs_opencv/
    ├── drs_3d_engine.py          # 3D Metric World Space Homography engine
    ├── physics_3d_predictor.py   # 3D Parabolic gravity height clearance model
    ├── ultraedge.py              # UltraEdge / Snickometer audio waveform simulator
    ├── hawk_eye_visualizer.py    # Broadcast-quality 4-panel 3D DRS TV card renderer
    ├── ai_verdict.py             # AI Natural Language verdict explainer & confidence scorer
    ├── stats_analyzer.py         # Delivery analytics (speed, swing, angle change)
    ├── frame_preprocessor.py     # CLAHE + Gaussian denoise preprocessing
    ├── confidence_scorer.py      # Weighted Kalman filter detection scorer
    ├── report_generator.py       # Exports full DRS analysis as drs_report.json
    ├── pipeline_logger.py        # Stage timing & metric logger
    ├── config.py                 # All tunable constants (colors, geometry, thresholds)
    ├── ball_detector.py          # HSV + contour + Hough ball detection
    ├── tracker.py                # Kalman-filter based multi-frame tracking
    ├── trajectory_predictor.py   # 2D piecewise-linear trajectory fitting
    ├── stump_zone.py             # Stump geometry + zone classification logic
    ├── visualizer.py             # Live tracking overlay drawing
    ├── generate_test_video.py    # Creates synthetic test footage
    └── main.py                   # Real DRS core pipeline orchestrator
```

---

## 🚀 Quick Start (No Video Needed)

### Option 1 — Web App (Recommended)

Visit **[https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)**, choose a synthetic outcome, and click **Run Hawk-Eye 3D Review**.

### Option 2 — Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/udbhav968-creator/opencv.git
cd opencv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask web app
python app.py
# Open http://localhost:5000 in your browser
```

### Option 3 — CLI Pipeline

```bash
cd drs_opencv

# Generate a synthetic test video
python generate_test_video.py --out sample_input.mp4 --outcome hitting

# Run the Real DRS pipeline
python main.py --input sample_input.mp4 --output_dir output
```

Outputs land in `output/`:
- `tracked_output.mp4` — annotated tracking video
- `drs_decision.png` — 4-panel broadcast TV Hawk-Eye graphic

---

## 📄 License

MIT License © [udbhav968-creator](https://github.com/udbhav968-creator)
