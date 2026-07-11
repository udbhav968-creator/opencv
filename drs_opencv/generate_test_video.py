"""
generate_test_video.py
------------------------
Generates a synthetic cricket-delivery video so the DRS pipeline can be
run and tested without needing real footage. Draws a green pitch, tan
pitch strip, stumps at both ends, and a red ball following a physically
plausible path: travels from the bowler's end towards the batsman,
bounces partway down (changing its lateral drift slightly, simulating
seam movement), then continues to an "impact" point near the batsman's
pad.

Usage:
    python generate_test_video.py --out sample_input.mp4 --outcome hitting
    python generate_test_video.py --out sample_input.mp4 --outcome missing
"""

import argparse
import cv2
import numpy as np
import config as cfg


def build_delivery_path(n_frames, start_x, pitch_x, end_x, bounce_frame):
    """
    Builds an (x, y) path for the ball across n_frames.
    y goes linearly from BOWLER_END_Y to BATSMAN_END_Y (constant pace).
    x is piecewise-linear: start_x -> pitch_x (at bounce_frame) -> end_x
    (at the final frame), simulating a straight line before the bounce
    and a (possibly) deviated straight line after -- exactly the shape
    trajectory_predictor.py is designed to recover.
    """
    ys = np.linspace(cfg.BOWLER_END_Y, cfg.BATSMAN_END_Y, n_frames)

    xs = np.zeros(n_frames)
    xs[:bounce_frame] = np.linspace(start_x, pitch_x, bounce_frame)
    xs[bounce_frame:] = np.linspace(pitch_x, end_x, n_frames - bounce_frame)

    return list(zip(xs, ys))


def render_frame(ball_pos, ball_radius=9, trail=None):
    frame = np.full((cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, 3), (30, 110, 40), dtype=np.uint8)

    # Pitch strip (tan)
    cv2.rectangle(
        frame,
        (cfg.FRAME_CENTER_X - 140, cfg.BOWLER_END_Y - 20),
        (cfg.FRAME_CENTER_X + 140, cfg.BATSMAN_END_Y + 40),
        (70, 150, 190), -1,
    )

    # Crease lines
    cv2.line(frame, (0, cfg.BOWLER_END_Y), (cfg.FRAME_WIDTH, cfg.BOWLER_END_Y), (200, 200, 200), 1)
    cv2.line(frame, (0, cfg.BATSMAN_END_Y), (cfg.FRAME_WIDTH, cfg.BATSMAN_END_Y), (200, 200, 200), 1)

    # Stumps (simple 3-line representation) at both ends
    for y_depth, w in [(cfg.BOWLER_END_Y, cfg.STUMPS_WIDTH_FAR), (cfg.BATSMAN_END_Y, cfg.STUMPS_WIDTH_NEAR)]:
        cx = cfg.FRAME_CENTER_X
        for dx in np.linspace(-w / 2, w / 2, 3):
            cv2.line(frame, (int(cx + dx), y_depth - 20), (int(cx + dx), y_depth), (210, 210, 210), 3)

    # Motion trail (faint, neutral gray so it doesn't get picked up by the
    # ball's colour mask -- broadcast graphics render trails in a distinct
    # colour from the ball itself for the same reason).
    if trail:
        for i, (tx, ty) in enumerate(trail[-15:]):
            alpha = (i + 1) / 15
            shade = int(60 + 60 * alpha)
            cv2.circle(frame, (int(tx), int(ty)), 2, (shade, shade, shade), -1)

    # Ball
    bx, by = ball_pos
    cv2.circle(frame, (int(bx), int(by)), ball_radius, (0, 0, 200), -1)
    cv2.circle(frame, (int(bx), int(by)), ball_radius, (0, 0, 90), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic DRS test video.")
    parser.add_argument("--out", default="sample_input.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames")
    parser.add_argument(
        "--outcome",
        choices=["hitting", "missing", "umpires_call"],
        default="hitting",
        help="What the final predicted trajectory should do to the stumps",
    )
    args = parser.parse_args()

    n_frames = args.frames
    bounce_frame = int(n_frames * 0.55)

    start_x = cfg.FRAME_CENTER_X - 15  # ball released just outside off-ish
    pitch_x = cfg.FRAME_CENTER_X - 5   # pitches close to in-line

    if args.outcome == "hitting":
        end_x = cfg.FRAME_CENTER_X + 2       # continues into the stumps
    elif args.outcome == "missing":
        end_x = cfg.FRAME_CENTER_X + 70      # drifts well outside off/leg
    else:  # umpires_call
        end_x = cfg.FRAME_CENTER_X + 30      # clips the edge of the stumps

    path = build_delivery_path(n_frames, start_x, pitch_x, end_x, bounce_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, cfg.FPS, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))

    trail = []
    for pos in path:
        frame = render_frame(pos, trail=trail)
        writer.write(frame)
        trail.append(pos)

    # Hold the last frame briefly (impact) so the pipeline has time to
    # register the final position clearly.
    for _ in range(5):
        writer.write(render_frame(path[-1], trail=trail))

    writer.release()
    print(f"Synthetic test video written to: {args.out}  ({n_frames + 5} frames, outcome={args.outcome})")


if __name__ == "__main__":
    main()
