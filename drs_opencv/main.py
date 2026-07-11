"""
main.py
-------
Runs the full DRS pipeline end-to-end on a video file:

    1. Read video frame by frame.
    2. Detect the ball each frame (ball_detector.py).
    3. Smooth/track it across frames (tracker.py).
    4. Fit the pitch point + predicted post-impact path (trajectory_predictor.py).
    5. Classify pitching / impact / wickets zones (stump_zone.py).
    6. Write an annotated tracking video AND a final decision-graphic image.

Usage:
    python main.py --input sample_input.mp4 --output_dir output
    python main.py --input sample_input.mp4 --color white   # for a white ball
"""

import argparse
import os
import cv2

import config as cfg
from ball_detector import BallDetector
from tracker import BallTracker
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

    detector = BallDetector(color_mode=color_mode)
    tracker = BallTracker()

    tracking_video_path = os.path.join(output_dir, "tracked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tracking_video_path, fourcc, src_fps, (src_w, src_h))

    frame_index = 0
    frames_with_ball = 0

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to the configured working resolution if needed, so the
        # pitch/stump geometry in config.py lines up with the frame.
        if (src_w, src_h) != (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT):
            frame = cv2.resize(frame, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))

        detection = detector.detect(frame)
        if detection is not None:
            frames_with_ball += 1

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

    # ---- Trajectory analysis & DRS decision ----
    valid_points = tracker.get_valid_trajectory_points()
    prediction = tp.predict_trajectory(valid_points)

    decision_image_path = os.path.join(output_dir, "drs_decision.png")

    if not prediction.has_prediction:
        print(
            "Not enough confident ball detections to compute a trajectory "
            "prediction. Try adjusting the HSV thresholds in config.py for "
            "your footage, or use --color white for a white ball."
        )
        return {
            "success": False,
            "tracking_video": tracking_video_path,
            "decision_image": None,
        }

    pitching_zone = stump_zone.classify_lateral_zone(*prediction.pitch_point)
    impact_zone = stump_zone.classify_lateral_zone(*prediction.impact_point)
    wicket_verdict = stump_zone.classify_wicket_hit(prediction.predicted_stump_x)

    canvas, final_call = visualizer.draw_decision_graphic(
        valid_points, prediction, pitching_zone, impact_zone, wicket_verdict
    )
    cv2.imwrite(decision_image_path, canvas)

    print("---- DRS RESULT ----")
    print(f"Pitching zone : {pitching_zone}")
    print(f"Impact zone   : {impact_zone}")
    print(f"Wickets       : {wicket_verdict}")
    print(f"Final call    : {final_call}")
    print(f"Decision graphic saved to: {decision_image_path}")

    return {
        "success": True,
        "tracking_video": tracking_video_path,
        "decision_image": decision_image_path,
        "pitching_zone": pitching_zone,
        "impact_zone": impact_zone,
        "wicket_verdict": wicket_verdict,
        "final_call": final_call,
    }


def main():
    parser = argparse.ArgumentParser(description="OpenCV DRS (Decision Review System) simulation pipeline.")
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
