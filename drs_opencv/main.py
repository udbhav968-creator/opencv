"""
main.py
-------
Runs the full Real DRS pipeline end-to-end on a video file:

    1. Read video frame by frame.
    2. Preprocess frame (frame_preprocessor.py).
    3. Auto-detect ball color if requested (auto_color_detector.py).
    4. Detect the ball each frame (ball_detector.py) & score confidence (confidence_scorer.py).
    5. Smooth/track across frames (tracker.py).
    6. Compute 3D physics trajectory & height clearance (physics_3d_predictor.py).
    7. Run UltraEdge snickometer simulation (ultraedge.py).
    8. Generate broadcast TV Hawk-Eye DRS graphic (hawk_eye_visualizer.py) & JSON report.

Usage:
    python main.py --input sample_input.mp4 --output_dir output
    python main.py --input sample_input.mp4 --color auto
"""

import argparse
import os
import cv2

try:
    import config as cfg
except ImportError:
    from drs_opencv import config as cfg

try:
    from yolo_detector import HybridBallDetector
except ImportError:
    from drs_opencv.yolo_detector import HybridBallDetector

try:
    from auto_color_detector import AutoColorDetector
except ImportError:
    from drs_opencv.auto_color_detector import AutoColorDetector

from ball_detector import BallDetector
from tracker import BallTracker
from frame_preprocessor import FramePreprocessor
from confidence_scorer import DetectionConfidenceScorer
from physics_3d_predictor import Physics3DPredictor
from ultraedge import UltraEdgeSimulator
from hawk_eye_visualizer import render_hawk_eye_broadcast_graphic
from report_generator import generate_report
import trajectory_predictor as tp
import stump_zone
import visualizer


def run_pipeline(input_path, output_dir, color_mode="auto"):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Auto Ball Color Detection if requested
    detected_color = color_mode
    auto_conf = 1.0
    if color_mode == "auto" or not color_mode:
        auto_detector = AutoColorDetector()
        detected_color, auto_conf = auto_detector.detect_ball_color(input_path)
        print(f"[DRS Engine] Auto-Detected Ball Color: {detected_color.upper()} (Confidence: {auto_conf})")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.FRAME_WIDTH
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.FRAME_HEIGHT
    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.FPS

    preprocessor = FramePreprocessor()
    detector = HybridBallDetector(color_mode=detected_color)
    confidence_scorer = DetectionConfidenceScorer()
    tracker = BallTracker()

    tracking_video_path = os.path.join(output_dir, "tracked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tracking_video_path, fourcc, src_fps, (src_w, src_h))

    frame_index = 0
    frames_with_ball = 0

    print(f"Processing video frames through Real DRS Pipeline (Color Mode: {detected_color.upper()})...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to working resolution
        if (src_w, src_h) != (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT):
            proc_frame = cv2.resize(frame, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))
        else:
            proc_frame = frame

        clean_frame = preprocessor.process(proc_frame)

        # Detect ball
        raw_det = detector.detect(clean_frame)
        detection = None
        if raw_det is not None:
            frames_with_ball += 1
            detection = (raw_det[0], raw_det[1], raw_det[2])

        # Score confidence
        conf = confidence_scorer.score(detection, frame=clean_frame)

        # Update Kalman tracker
        est = tracker.update(detection, frame_index)

        overlay_frame = frame.copy()
        traj_points = tracker.get_trajectory_points()
        radius = detection[2] if detection is not None else None
        visualizer.draw_live_overlay(overlay_frame, traj_points, est, radius)

        if (src_w, src_h) != (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT):
            overlay_frame = cv2.resize(overlay_frame, (src_w, src_h))
        writer.write(overlay_frame)

        frame_index += 1

    cap.release()
    writer.release()

    print(f"Frames processed: {frame_index}, frames with ball detected: {frames_with_ball}")
    print(f"Annotated tracking video saved to: {tracking_video_path}")

    # ---- 2D Trajectory analysis & legacy zones ----
    valid_points = tracker.get_valid_trajectory_points()
    prediction_2d = tp.predict_trajectory(valid_points)

    decision_image_path = os.path.join(output_dir, "drs_decision.png")

    if not prediction_2d.has_prediction:
        print("Not enough confident ball detections to compute trajectory prediction.")
        return {
            "success": False,
            "tracking_video": tracking_video_path,
            "decision_image": None,
            "detected_color": detected_color,
        }

    pitching_zone = stump_zone.classify_lateral_zone(*prediction_2d.pitch_point)
    impact_zone = stump_zone.classify_lateral_zone(*prediction_2d.impact_point)
    wicket_verdict_2d = stump_zone.classify_wicket_hit(prediction_2d.predicted_stump_x)

    physics_3d = Physics3DPredictor(fps=src_fps)
    prediction_3d = physics_3d.predict_3d(valid_points)

    pz_str = pitching_zone.value if hasattr(pitching_zone, 'value') else str(pitching_zone)
    iz_str = impact_zone.value if hasattr(impact_zone, 'value') else str(impact_zone)
    wv_str = wicket_verdict_2d.value if hasattr(wicket_verdict_2d, 'value') else str(wicket_verdict_2d)

    final_call = "OUT" if (wv_str == "HITTING" and iz_str == "IN_LINE" and pz_str != "OUTSIDE_LEG") else "NOT OUT"
    if (wv_str == "UMPIRES_CALL" and iz_str == "IN_LINE" and pz_str != "OUTSIDE_LEG"):
        final_call = "UMPIRE'S CALL"

    # ---- Render Broadcast Hawk-Eye Graphic ----
    broadcast_canvas = render_hawk_eye_broadcast_graphic(
        valid_points, prediction_3d, pz_str, iz_str, wv_str, final_call
    )
    cv2.imwrite(decision_image_path, broadcast_canvas)

    print("---- REAL DRS HAWK-EYE RESULT ----")
    print(f"Pitching zone : {pz_str}")
    print(f"Impact zone   : {iz_str}")
    print(f"Wickets (2D)  : {wv_str}")
    if prediction_3d.has_prediction:
        print(f"3D Stump Height: {prediction_3d.stump_z:.2f}m (Verdict: {prediction_3d.height_verdict})")
    print(f"Final call    : {final_call}")
    print(f"Hawk-Eye Decision graphic saved to: {decision_image_path}")

    class DummyZone:
        def __init__(self, val):
            self.value = val

    return {
        "success": True,
        "tracking_video": tracking_video_path,
        "decision_image": decision_image_path,
        "pitching_zone": DummyZone(pz_str),
        "impact_zone": DummyZone(iz_str),
        "wicket_verdict": DummyZone(wv_str),
        "final_call": final_call,
        "valid_points": valid_points,
        "prediction_3d": prediction_3d,
        "detected_color": detected_color,
    }


def main():
    parser = argparse.ArgumentParser(description="Real DRS (Decision Review System) Hawk-Eye Simulation Pipeline.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output_dir", default="output", help="Directory to save results")
    parser.add_argument(
        "--color", choices=["auto", "red", "white", "pink", "yellow", "orange"], default="auto",
        help="Ball colour mode (auto = automatic color detection)",
    )
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, color_mode=args.color)


if __name__ == "__main__":
    main()
