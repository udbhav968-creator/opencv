# 🏏 OpenCV DRS — Cricket Decision Review System

> A complete end-to-end Computer Vision + AI web application that simulates the cricket Decision Review System (DRS) — **detect → track → predict → decide** — and renders a broadcast-style review graphic, plus an annotated tracking video.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?style=flat-square&logo=opencv)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=flat-square&logo=vercel)](https://opencv-lyart.vercel.app)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 🌐 Live Demo

**[https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)**

---

## ✨ Features

- 🎯 **Ball Detection** — HSV colour thresholding + contour circularity checks + Hough-circle fallback
- 🔄 **Kalman Filter Tracking** — smooth multi-frame tracking through occlusions (bat/pad)
- 📐 **Trajectory Prediction** — piecewise-linear regression to find the pitch (bounce) point and project post-impact path to stumps
- 🟢 **Zone Classification** — Pitching / Impact (Outside Leg / In-line / Outside Off) and Wickets (Hitting / Umpire's Call / Missing)
- 📹 **Annotated Video Output** — live tracking overlay video saved as `.mp4`
- 🖼️ **DRS Decision Graphic** — Hawk-Eye-style broadcast image with coloured stumps, dotted predicted path, and final call banner
- 🌐 **Flask Web App** — upload a video or generate synthetic test footage right in your browser
- 🚀 **Vercel Deployment** — production-ready serverless Python deployment

---

## 🏗️ Project Structure

```
opencv/
├── app.py                        # Flask web application (main entry point)
├── requirements.txt              # Python dependencies
├── vercel.json                   # Vercel deployment configuration
├── templates/
│   └── index.html                # Dark-mode web UI
└── drs_opencv/
    ├── config.py                 # All tunable constants (colors, geometry, thresholds)
    ├── ball_detector.py          # HSV + contour + Hough ball detection
    ├── tracker.py                # Kalman-filter based multi-frame tracking
    ├── trajectory_predictor.py   # Pitch-point fitting + forward projection
    ├── stump_zone.py             # Stump geometry + zone classification logic
    ├── visualizer.py             # Drawing: live overlay + final decision graphic
    ├── generate_test_video.py    # Creates synthetic test footage (no real video needed)
    ├── main.py                   # Core pipeline orchestrator
    └── README.md                 # Detailed module-level documentation
```

---

## 🚀 Quick Start (No Video Needed)

The project ships with a **synthetic footage generator**, so you can test the pipeline immediately without any real cricket video.

### Option 1 — Web App (Recommended)

Visit **[https://opencv-lyart.vercel.app](https://opencv-lyart.vercel.app)**, choose a synthetic outcome, and click **Process Delivery**.

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

### Option 3 — CLI (Original Pipeline)

```bash
cd drs_opencv

# Generate a synthetic test video
python generate_test_video.py --out sample_input.mp4 --outcome hitting

# Run the DRS pipeline
python main.py --input sample_input.mp4 --output_dir output

# Try other outcomes
python generate_test_video.py --out sample.mp4 --outcome missing
python generate_test_video.py --out sample.mp4 --outcome umpires_call
```

Outputs land in `output/`:
- `tracked_output.mp4` — annotated tracking video
- `drs_decision.png` — broadcast-style DRS graphic

---

## ⚙️ Configuration

Tweak `drs_opencv/config.py` to adapt to your footage:

| Parameter | Description |
|-----------|-------------|
| `RED_BALL_LOWER/UPPER` | HSV range for red ball detection |
| `WHITE_BALL_LOWER/UPPER` | HSV range for white ball detection |
| `MIN_BALL_RADIUS` / `MAX_BALL_RADIUS` | Expected ball size in pixels |
| `BOWLER_END_Y`, `BATSMAN_END_Y` | Pitch boundary pixel positions |
| `STUMPS_WIDTH_FAR/NEAR` | Stump geometry for your camera angle |

---

## 🧠 How It Works

### 1. Ball Detection (`ball_detector.py`)
- Converts each frame to **HSV colour space**
- Applies a colour mask (red or white) to isolate ball-coloured regions
- Uses **contour analysis** (area + circularity) to find ball candidates
- Falls back to **Hough Circle Transform** for difficult frames

### 2. Tracking (`tracker.py`)
- A **Kalman Filter** smooths noisy detections across frames
- Coasts through brief occlusions (bat/pad blocking the ball)
- Maintains a trajectory history of smoothed positions

### 3. Trajectory Prediction (`trajectory_predictor.py`)
- Runs a **piecewise-linear regression search** across all plausible split points
- Finds the pitch (bounce) point as the split that minimises total fitting error
- Projects the **pre-impact straight line** forward to the stump plane

### 4. Zone Classification (`stump_zone.py`)
- Classifies pitching zone: `OUTSIDE_LEG` / `IN_LINE` / `OUTSIDE_OFF`
- Classifies impact zone the same way
- Classifies wickets: `HITTING` / `UMPIRES_CALL` / `MISSING`
- Combines all three for the final **OUT / NOT OUT / UMPIRE'S CALL** decision

### 5. Visualisation (`visualizer.py`)
- Draws the live tracking overlay frame-by-frame
- Renders a **broadcast-style decision graphic** with green pitch, dotted predicted path, and coloured stumps

---

## 🌐 Web Application

The Flask web app (`app.py`) wraps the entire pipeline into an interactive UI:

- **Upload** your own `.mp4` cricket delivery video, OR
- **Generate** a synthetic test video (Hitting / Missing / Umpire's Call)
- Choose **ball colour** (Red for Test Match, White for Limited Overs)
- Get back:
  - The **annotated tracking video**
  - The **DRS decision graphic**
  - Pitching zone, Impact zone, Wickets verdict, and Final call

---

## 🚢 Deployment

This app is deployed on **Vercel** using the `@vercel/python` serverless runtime.

```json
// vercel.json
{
  "builds": [{ "src": "app.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "app.py" }]
}
```

> **Note:** File outputs are written to `/tmp/` on Vercel (serverless ephemeral storage). For persistent storage, integrate a cloud bucket (e.g. AWS S3 or Cloudflare R2).

---

## ⚠️ Limitations

This is an **educational simulation**, not a certified DRS system:

- **Single camera, 2D only** — no height/depth triangulation
- **No handedness detection** — off-side/leg-side is fixed to ±x axis
- **Straight-line post-impact projection** — no physics modelling for spin/seam after bounce
- **Colour-based detection** — needs threshold tuning for real broadcast footage

---

## 🔮 Future Enhancements

- [ ] **YOLO-based detection** — replace HSV with a trained object detector for robustness
- [ ] **Parabolic vertical modelling** — reason about ball height over stumps using gravity
- [ ] **Multi-camera 3D triangulation** — true Hawk-Eye-style 3D tracking
- [ ] **Real-time webcam support** — live delivery analysis
- [ ] **AI Verdict Explainer** — natural language summary of the DRS decision

---

## 🤝 Contributing

Contributions are welcome! Open a pull request or an issue.

---

## 📄 License

MIT License © [udbhav968-creator](https://github.com/udbhav968-creator)

---

## 👤 Author

**GitHub:** [https://github.com/udbhav968-creator](https://github.com/udbhav968-creator)
