"""
visualizer.py
-------------
Drawing helpers:
  - draw_live_overlay(): per-frame ball marker + trailing trajectory line,
    used while writing the annotated tracking video.
  - draw_stumps(): perspective-correct stump box at any depth.
  - draw_decision_graphic(): the final broadcast-style "ball tracking"
    review card, with pitching marker, predicted dotted path, stumps,
    and the PITCHING / IMPACT / WICKETS verdict text.
"""

import cv2
import numpy as np
import config as cfg
import stump_zone


def draw_stumps(frame, y_depth, color=cfg.COLOR_STUMPS, thickness=2):
    x1, y1, x2, y2 = stump_zone.get_stump_box(y_depth)
    n_stumps = 3
    stump_w = max(1, (x2 - x1) // (n_stumps * 2))
    xs = np.linspace(x1, x2, n_stumps)
    for x in xs:
        cv2.line(frame, (int(x), y1), (int(x), y2), color, thickness)
    # bails
    cv2.line(frame, (x1, y1), (x2, y1), color, max(1, thickness - 1))


def draw_pitch_guides(frame):
    """Draws the crease lines at bowler & batsman ends for context."""
    cv2.line(frame, (0, cfg.BOWLER_END_Y), (cfg.FRAME_WIDTH, cfg.BOWLER_END_Y),
              (90, 90, 90), 1)
    cv2.line(frame, (0, cfg.BATSMAN_END_Y), (cfg.FRAME_WIDTH, cfg.BATSMAN_END_Y),
              (90, 90, 90), 1)
    draw_stumps(frame, cfg.BOWLER_END_Y)
    draw_stumps(frame, cfg.BATSMAN_END_Y)


def draw_live_overlay(frame, trajectory_points, current_point, current_radius=None):
    """Draws the trailing path + current ball marker onto a live frame."""
    draw_pitch_guides(frame)

    pts = [(int(x), int(y)) for x, y in trajectory_points]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], cfg.COLOR_TRAJECTORY, 2)

    if current_point is not None:
        cx, cy = int(current_point[0]), int(current_point[1])
        r = int(current_radius) if current_radius else 6
        cv2.circle(frame, (cx, cy), r, cfg.COLOR_WHITE, -1)
        cv2.circle(frame, (cx, cy), r, cfg.COLOR_BLACK, 2)

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
    Builds the final still-frame "review card" summarizing the DRS
    decision, in the style of broadcast ball-tracking graphics.
    """
    canvas = np.full((cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, 3), (30, 100, 40), dtype=np.uint8)

    # Pitch strip
    cv2.rectangle(
        canvas,
        (cfg.FRAME_CENTER_X - 140, cfg.BOWLER_END_Y - 20),
        (cfg.FRAME_CENTER_X + 140, cfg.BATSMAN_END_Y + 20),
        (70, 150, 190), -1,
    )
    draw_pitch_guides(canvas)

    # Actual tracked path (solid)
    pts = [(int(x), int(y)) for x, y in valid_points]
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], cfg.COLOR_TRAJECTORY, 3)

    if prediction.has_prediction:
        # Pitching marker
        px, py = prediction.pitch_point
        cv2.circle(canvas, (int(px), int(py)), 7, cfg.COLOR_PITCH_MARKER, -1)
        cv2.circle(canvas, (int(px), int(py)), 7, cfg.COLOR_BLACK, 2)
        cv2.putText(canvas, "PITCH", (int(px) + 10, int(py) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg.COLOR_WHITE, 1, cv2.LINE_AA)

        # Impact marker
        ix, iy = prediction.impact_point
        cv2.drawMarker(canvas, (int(ix), int(iy)), cfg.COLOR_YELLOW,
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=14, thickness=2)
        cv2.putText(canvas, "IMPACT", (int(ix) + 10, int(iy) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg.COLOR_WHITE, 1, cv2.LINE_AA)

        # Predicted dotted path from impact to stumps
        predicted_end = (prediction.predicted_stump_x, cfg.BATSMAN_END_Y)
        verdict_color = {
            "HITTING": cfg.COLOR_GREEN,
            "UMPIRES_CALL": cfg.COLOR_YELLOW,
            "MISSING": cfg.COLOR_RED,
        }[wicket_verdict]

        _dotted_line(canvas, (ix, iy), predicted_end, verdict_color, thickness=3)

        # Highlight stumps at batsman end according to verdict
        draw_stumps(canvas, cfg.BATSMAN_END_Y, color=verdict_color, thickness=3)

    # Text panel
    panel_y = 30
    cv2.putText(canvas, "DRS - BALL TRACKING (SIMULATION)", (20, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, cfg.COLOR_WHITE, 2, cv2.LINE_AA)

    lines = [
        f"PITCHING: {pitching_zone}",
        f"IMPACT:   {impact_zone}",
        f"WICKETS:  {wicket_verdict}",
    ]
    verdict_line_color = {
        "HITTING": cfg.COLOR_GREEN,
        "UMPIRES_CALL": cfg.COLOR_YELLOW,
        "MISSING": cfg.COLOR_RED,
    }[wicket_verdict]
    colors = [cfg.COLOR_WHITE, cfg.COLOR_WHITE, verdict_line_color]

    y0 = cfg.FRAME_HEIGHT - 90
    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(canvas, line, (20, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    final_call = "OUT" if wicket_verdict == "HITTING" and impact_zone == "IN_LINE" and pitching_zone != "OUTSIDE_LEG" else "NOT OUT"
    if wicket_verdict == "UMPIRES_CALL" and impact_zone == "IN_LINE" and pitching_zone != "OUTSIDE_LEG":
        final_call = "UMPIRE'S CALL"

    call_color = {
        "OUT": cfg.COLOR_RED,
        "NOT OUT": cfg.COLOR_GREEN,
        "UMPIRE'S CALL": cfg.COLOR_YELLOW,
    }[final_call]
    cv2.putText(canvas, final_call, (cfg.FRAME_WIDTH - 260, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, call_color, 3, cv2.LINE_AA)

    return canvas, final_call
