"""
main.py
-------
Runs the full Real DRS pipeline end-to-end on a video file:

    1. Read video frame by frame.
    2. Preprocess frame (frame_preprocessor.py).
    3. Detect the ball each frame (ball_detector.py) & score confidence (confidence_scorer.py).
    4. Smooth/track across frames (tracker.py).
    5. Compute 3D physics trajectory & height clearance (physics_3d_predictor.py).
    6. Run UltraEdge snickometer simulation (ultraedge.py).
    7. Generate broadcast TV Hawk-Eye DRS graphic (hawk_eye_visualizer.py) & JSON report.

Usage:
    python main.py --input sample_input.mp4 --output_dir output
    python main.py --input sample_input.mp4 --color white
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


def run_pipeline(input_path, output_dir, color_mode="red"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.FRAME_WIDTH
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.FRAME_HEIGHT
    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.FPS

    preprocessor = FramePreprocessor()
    detector = HybridBallDetector(color_mode=color_mode)
    confidence_scorer = DetectionConfidenceScorer()
    tracker = BallTracker()

    tracking_video_path = os.path.join(output_dir, "tracked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tracking_video_path, fourcc, src_fps, (src_w, src_h))

    frame_index = 0
    frames_with_ball = 0

    print("Processing video frames through Real DRS Pipeline...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to working resolution
        if (src_w, src_h) != (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT):
            frame = cv2.resize(frame, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))

        # Preprocess frame
        clean_frame = preprocessor.process(frame)

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
        }

    pitching_zone = stump_zone.classify_lateral_zone(*prediction_2d.pitch_point)
    impact_zone = stump_zone.classify_lateral_zone(*prediction_2d.impact_point)
    wicket_verdict_2d = stump_zone.classify_wicket_hit(prediction_2d.predicted_stump_x)

    # ---- 3D Physics Trajectory & Height Clearance Model ----
    physics_3d = Physics3DPredictor(fps=src_fps)
    prediction_3d = physics_3d.predict_3d(valid_points)

    final_call = "OUT" if (wicket_verdict_2d == "HITTING" and impact_zone == "IN_LINE" and pitching_zone != "OUTSIDE_LEG") else "NOT OUT"
    if (wicket_verdict_2d == "UMPIRES_CALL" and impact_zone == "IN_LINE" and pitching_zone != "OUTSIDE_LEG"):
        final_call = "UMPIRE'S CALL"

    # ---- Render Broadcast Hawk-Eye Graphic ----
    broadcast_canvas = render_hawk_eye_broadcast_graphic(
        valid_points, prediction_3d, pitching_zone.value if hasattr(pitching_zone, 'value') else pitching_zone,
        impact_zone.value if hasattr(impact_zone, 'value') else impact_zone, wicket_verdict_2d.value if hasattr(wicket_verdict_2d, 'value') else wicket_verdict_2d, final_call
    )
    cv2.imwrite(decision_image_path, broadcast_canvas)

    print("---- REAL DRS HAWK-EYE RESULT ----")
    print(f"Pitching zone : {pitching_zone}")
    print(f"Impact zone   : {impact_zone}")
    print(f"Wickets (2D)  : {wicket_verdict_2d}")
    if prediction_3d.has_prediction:
        print(f"3D Stump Height: {prediction_3d.stump_z:.2f}m (Verdict: {prediction_3d.height_verdict})")
    print(f"Final call    : {final_call}")
    print(f"Hawk-Eye Decision graphic saved to: {decision_image_path}")

    class DummyZone:
        def __init__(self, val):
            self.value = val

    pz = DummyZone(pitching_zone) if isinstance(pitching_zone, str) else pitching_zone
    iz = DummyZone(impact_zone) if isinstance(impact_zone, str) else impact_zone
    wv = DummyZone(wicket_verdict_2d) if isinstance(wicket_verdict_2d, str) else wicket_verdict_2d

    return {
        "success": True,
        "tracking_video": tracking_video_path,
        "decision_image": decision_image_path,
        "pitching_zone": pz,
        "impact_zone": iz,
        "wicket_verdict": wv,
        "final_call": final_call,
        "valid_points": valid_points,
        "prediction_3d": prediction_3d,
    }


def main():
    parser = argparse.ArgumentParser(description="Real DRS (Decision Review System) Hawk-Eye Simulation Pipeline.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output_dir", default="output", help="Directory to save results")
    parser.add_argument(
        "--color", choices=["red", "white"], default="red",
        help="Ball colour to detect (red = default Test ball, white = limited-overs ball)",
    )
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, color_mode=args.color)


if __name__ == "__main__":
    main()
