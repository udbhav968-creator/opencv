# visualizer.py
"""
visualizer.py
-------------
Real-World TV Broadcast Visualizer Engine for Hawk-Eye 3D DRS.
Includes:
  - Red Neon Glowing Trajectory Trail (Glow Ribbon) behind moving ball.
  - Predicted Flight Extension Path (Dotted Neon Line past impact to stumps).
  - Pitch Bounce Spot & Leg Pad Impact Markers with 3D Ring Effects.
  - Broadcast TV HUD Banner Overlay (Pitching / Impact / Stumps / Final Call).
"""

import cv2
import numpy as np
try:
    import config as cfg
    import stump_zone
except ImportError:
    from drs_opencv import config as cfg
    from drs_opencv import stump_zone


def draw_stumps(frame, y_depth, color=cfg.COLOR_STUMPS, thickness=2):
    x1, y1, x2, y2 = stump_zone.get_stump_box(y_depth)
    n_stumps = 3
    stump_w = max(1, (x2 - x1) // (n_stumps * 2))
    xs = np.linspace(x1, x2, n_stumps)
    for x in xs:
        cv2.line(frame, (int(x), y1), (int(x), y2), color, thickness)
    # Bails
    cv2.line(frame, (x1, y1), (x2, y1), color, max(1, thickness - 1))


def draw_pitch_guides(frame):
    """Draws the crease lines at bowler & batsman ends for context."""
    cv2.line(frame, (0, cfg.BOWLER_END_Y), (cfg.FRAME_WIDTH, cfg.BOWLER_END_Y), (100, 100, 100), 1)
    cv2.line(frame, (0, cfg.BATSMAN_END_Y), (cfg.FRAME_WIDTH, cfg.BATSMAN_END_Y), (100, 100, 100), 1)
    draw_stumps(frame, cfg.BOWLER_END_Y, color=(200, 200, 200), thickness=1)
    draw_stumps(frame, cfg.BATSMAN_END_Y, color=(250, 204, 21), thickness=2)


def draw_live_overlay(frame, trajectory_points, current_point, current_radius=None, pitching_zone=None, impact_zone=None, wicket_verdict=None, final_call=None):
    """
    Draws Real-World TV Broadcast Trajectory Trail & HUD directly onto live video frames.
    """
    h, w, _ = frame.shape
    draw_pitch_guides(frame)

    pts = [(int(x), int(y)) for x, y in trajectory_points]
    n_pts = len(pts)

    # 1. Draw Red Neon Glowing Trajectory Ribbon
    for i in range(1, n_pts):
        alpha = float(i) / max(1, n_pts)
        thickness = max(2, int(6 * alpha))
        
        # Red Glow Outer Ring
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness + 4, cv2.LINE_AA)
        # Red Inner Ribbon
        cv2.line(frame, pts[i - 1], pts[i], (50, 50, 255), thickness, cv2.LINE_AA)
        # Bright Yellow Core
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Current Ball Outer Halo & Pulse Ring
    if current_point is not None:
        cx, cy = int(current_point[0]), int(current_point[1])
        r = int(current_radius) if current_radius else 8
        
        # Red Glowing Ball Outer Ring
        cv2.circle(frame, (cx, cy), r + 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r + 2, (50, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)

    # 3. TV Broadcast Glassmorphic HUD Banner
    if final_call:
        banner_h = 70
        banner_w = w - 40
        overlay = frame.copy()
        
        # Dark semi-transparent HUD background
        cv2.rectangle(overlay, (20, 20), (20 + banner_w, 20 + banner_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (20, 20), (20 + banner_w, 20 + banner_h), (56, 189, 248), 2)

        # Broadcast Text
        cv2.putText(frame, "ICC REAL DRS HAWK-EYE 3D BROADCAST", (35, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 204, 21), 2, cv2.LINE_AA)
        
        hud_text = f"PITCHING: {pitching_zone or 'IN LINE'}  |  IMPACT: {impact_zone or 'IN LINE'}  |  WICKETS: {wicket_verdict or 'HITTING'}"
        cv2.putText(frame, hud_text, (35, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 250, 252), 1, cv2.LINE_AA)

        call_color = (0, 0, 255) if final_call == "OUT" else (0, 255, 0) if final_call == "NOT OUT" else (0, 255, 255)
        cv2.putText(frame, final_call, (w - 180, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, call_color, 3, cv2.LINE_AA)

    return frame


def _dotted_line(frame, pt1, pt2, color, thickness=2, gap=8):
    pt1 = np.array(pt1, dtype=float)
    pt2 = np.array(pt2, dtype=float)
    dist = np.linalg.norm(pt2 - pt1)
    if dist == 0:
        return
    n_dots = max(1, int(dist / gap))
    for i in range(n_dots + 1):
        t = i / n_dots
        p = pt1 + t * (pt2 - pt1)
        cv2.circle(frame, (int(p[0]), int(p[1])), thickness, color, -1)


def draw_decision_graphic(valid_points, prediction, pitching_zone, impact_zone, wicket_verdict):
    """
    Builds the final still-frame broadcast "ball tracking" review card.
    """
    canvas = np.full((cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, 3), (15, 23, 42), dtype=np.uint8)

    # Turf pitch strip
    cv2.rectangle(
        canvas,
        (cfg.FRAME_CENTER_X - 140, cfg.BOWLER_END_Y - 20),
        (cfg.FRAME_CENTER_X + 140, cfg.BATSMAN_END_Y + 20),
        (20, 83, 45), -1
    )
    draw_pitch_guides(canvas)

    # Actual Red Neon Flight Path
    pts = [(int(x), int(y)) for x, y in valid_points]
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], (0, 0, 255), 4, cv2.LINE_AA)
        cv2.line(canvas, pts[i - 1], pts[i], (50, 180, 255), 2, cv2.LINE_AA)

    if prediction.has_prediction:
        # Pitching Bounce Spot Ring
        px, py = prediction.pitch_point
        cv2.circle(canvas, (int(px), int(py)), 9, (250, 204, 21), -1)
        cv2.circle(canvas, (int(px), int(py)), 14, (56, 189, 248), 2, cv2.LINE_AA)
        cv2.putText(canvas, "PITCH BOUNCE", (int(px) + 12, int(py) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 204, 21), 2, cv2.LINE_AA)

        # Pad Impact Spot Ring
        ix, iy = prediction.impact_point
        cv2.drawMarker(canvas, (int(ix), int(iy)), (0, 255, 255),
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=3)
        cv2.putText(canvas, "PAD IMPACT", (int(ix) + 12, int(iy) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)

        # Projected Dotted Extension Path to Stumps
        predicted_end = (prediction.predicted_stump_x, cfg.BATSMAN_END_Y)
        verdict_color = (0, 255, 0) if wicket_verdict == "HITTING" else (0, 255, 255) if wicket_verdict == "UMPIRES_CALL" else (0, 0, 255)

        _dotted_line(canvas, (ix, iy), predicted_end, verdict_color, thickness=4)
        draw_stumps(canvas, cfg.BATSMAN_END_Y, color=verdict_color, thickness=3)

    # Broadcast Text Panel
    panel_y = 35
    cv2.putText(canvas, "ICC REAL DRS HAWK-EYE 3D BROADCAST", (20, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 204, 21), 2, cv2.LINE_AA)

    lines = [
        f"PITCHING: {pitching_zone}",
        f"IMPACT:   {impact_zone}",
        f"WICKETS:  {wicket_verdict}",
    ]
    verdict_line_color = (0, 255, 0) if wicket_verdict == "HITTING" else (0, 255, 255) if wicket_verdict == "UMPIRES_CALL" else (0, 0, 255)
    colors = [(255, 255, 255), (255, 255, 255), verdict_line_color]

    y0 = cfg.FRAME_HEIGHT - 90
    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(canvas, line, (20, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    final_call = "OUT" if wicket_verdict == "HITTING" and impact_zone == "IN_LINE" and pitching_zone != "OUTSIDE_LEG" else "NOT OUT"
    if wicket_verdict == "UMPIRES_CALL" and impact_zone == "IN_LINE" and pitching_zone != "OUTSIDE_LEG":
        final_call = "UMPIRE'S CALL"

    call_color = (0, 0, 255) if final_call == "OUT" else (0, 255, 0) if final_call == "NOT OUT" else (0, 255, 255)
    cv2.putText(canvas, final_call, (cfg.FRAME_WIDTH - 240, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, call_color, 3, cv2.LINE_AA)

    return canvas, final_call
