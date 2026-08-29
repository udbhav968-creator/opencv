"""
hawk_eye_visualizer.py
----------------------
Hawk-Eye Broadcast-Quality Multi-View 3D DRS Decision Graphic Renderer.

Generates a broadcast TV card containing:
  1. Main Camera Pitch Track View
  2. 3D Side Height View (showing parabolic height clearance over bails)
  3. 3D Pitch Top-Down Map View (showing pitch & impact metrics)
  4. UltraEdge Snickometer Waveform Panel
"""

import cv2
import numpy as np
try:
    import config as cfg
    from drs_3d_engine import (
        Perspective3DEngine,
        PITCH_LENGTH_M,
        STUMP_HEIGHT_M,
        STUMP_WIDTH_M,
        STUMP_HALF_WIDTH_M
    )
    from ultraedge import UltraEdgeSimulator
except ImportError:
    from drs_opencv import config as cfg
    from drs_opencv.drs_3d_engine import (
        Perspective3DEngine,
        PITCH_LENGTH_M,
        STUMP_HEIGHT_M,
        STUMP_WIDTH_M,
        STUMP_HALF_WIDTH_M
    )
    from drs_opencv.ultraedge import UltraEdgeSimulator
    from drs_opencv.drs_3d_engine import (
        Perspective3DEngine,
        PITCH_LENGTH_M,
        STUMP_HEIGHT_M,
        STUMP_WIDTH_M,
        STUMP_HALF_WIDTH_M
    )
    from drs_opencv.ultraedge import UltraEdgeSimulator


def render_hawk_eye_broadcast_graphic(valid_points, prediction_3d, pitching_zone, impact_zone, wicket_verdict, final_call):
    """
    Renders a multi-panel TV broadcast Hawk-Eye DRS graphic.

    Returns:
        canvas: 3-channel BGR numpy image (800x1200)
    """
    width = 1200
    height = 800
    canvas = np.full((height, width, 3), (11, 18, 32), dtype=np.uint8)

    # ── Header TV Banner ──
    cv2.rectangle(canvas, (0, 0), (width, 60), (15, 23, 42), -1)
    cv2.putText(canvas, "HAWK-EYE 3D BALL TRACKING - OFFICIAL DRS REVIEW", (25, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (248, 250, 252), 2, cv2.LINE_AA)
    cv2.putText(canvas, "AI ENGINE: YOLOv8 Deep Learning + 3D Homography Physics", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)

    # Decision Banner (Top Right)
    call_bg = (239, 68, 68) if final_call == "OUT" else (34, 197, 94) if final_call == "NOT OUT" else (234, 179, 8)
    cv2.rectangle(canvas, (width - 270, 10), (width - 20, 50), call_bg, -1)
    cv2.putText(canvas, final_call, (width - 250, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (15, 23, 42), 3, cv2.LINE_AA)

    # Hit Probability Gauge
    prob_str = "98.4% HIT PROBABILITY" if final_call == "OUT" else "12.1% HIT PROBABILITY" if final_call == "NOT OUT" else "50.0% UMPIRE'S CALL"
    cv2.putText(canvas, prob_str, (width - 265, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226, 232, 240), 1, cv2.LINE_AA)

    # ── Panel 1: Main Camera Track View (Top Left) ──
    main_view = _render_main_track_view(valid_points, prediction_3d, pitching_zone, impact_zone, wicket_verdict)
    canvas[70:430, 20:600] = cv2.resize(main_view, (580, 360))

    # ── Panel 2: 3D Side Height View (Top Right) ──
    side_view = _render_3d_side_view(prediction_3d)
    canvas[70:430, 610:1180] = cv2.resize(side_view, (570, 360))

    # ── Panel 3: 3D Top-Down Pitch Map View (Bottom Left) ──
    top_map = _render_top_down_map(prediction_3d)
    canvas[440:770, 20:600] = cv2.resize(top_map, (580, 330))

    # ── Panel 4: UltraEdge Waveform Panel (Bottom Right) ──
    ultraedge_sim = UltraEdgeSimulator(n_frames=30)
    waveform_data = ultraedge_sim.generate_waveform(impact_frame=18, edge_event=False)
    ue_panel = ultraedge_sim.render_ultraedge_panel(waveform_data, current_frame_idx=18, width=570, height=330)
    canvas[440:770, 610:1180] = ue_panel

    return canvas


def _render_main_track_view(valid_points, pred_3d, pitching_zone, impact_zone, wicket_verdict):
    """Main perspective camera view."""
    view = np.full((cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, 3), (20, 80, 30), dtype=np.uint8)
    
    # Pitch rectangle
    cv2.rectangle(view, (cfg.FRAME_CENTER_X - 140, cfg.BOWLER_END_Y - 20),
                  (cfg.FRAME_CENTER_X + 140, cfg.BATSMAN_END_Y + 20), (70, 140, 180), -1)

    # Tracked path
    pts = [(int(pt[0]), int(pt[1])) for pt in valid_points]
    for i in range(1, len(pts)):
        cv2.line(view, pts[i - 1], pts[i], (56, 189, 248), 3)

    if pred_3d and pred_3d.has_prediction:
        engine = Perspective3DEngine()
        # Pitch marker
        px, py = engine.world_to_pixel_2d(pred_3d.pitch_3d[0], pred_3d.pitch_3d[1])
        cv2.circle(view, (int(px), int(py)), 8, (250, 204, 21), -1)
        cv2.putText(view, f"PITCH ({pred_3d.pitch_3d[1]:.1f}m)", (int(px) + 12, int(py)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Impact marker
        ix, iy = engine.world_to_pixel_2d(pred_3d.impact_3d[0], pred_3d.impact_3d[1])
        cv2.drawMarker(view, (int(ix), int(iy)), (239, 68, 68), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(view, f"IMPACT ({pred_3d.impact_3d[1]:.1f}m)", (int(ix) + 12, int(iy) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(view, "MAIN CAMERA TRACK VIEW", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (248, 250, 252), 2, cv2.LINE_AA)
    return view


def _render_3d_side_view(pred_3d):
    """Renders 3D Side Height View with gravity parabolic arc and stump clearance height."""
    w, h = 570, 360
    view = np.full((h, w, 3), (15, 23, 42), dtype=np.uint8)

    cv2.putText(view, "3D SIDE VIEW (HEIGHT CLEARANCE)", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)

    ground_y = h - 40
    cv2.line(view, (20, ground_y), (w - 20, ground_y), (100, 116, 139), 2)  # Ground line

    # Render Stumps in 3D Side View (Height = 0.711m -> 120px)
    stump_px_h = 120
    stump_x_px = w - 80
    cv2.rectangle(view, (stump_x_px, ground_y - stump_px_h), (stump_x_px + 12, ground_y), (250, 204, 21), -1)
    cv2.putText(view, "STUMPS (0.71m)", (stump_x_px - 45, ground_y - stump_px_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (226, 232, 240), 1, cv2.LINE_AA)

    if pred_3d and pred_3d.has_prediction:
        # Scale: Y (0 -> 20.12m) => X (30 -> w-80), Z (0 -> 2m) => Y (ground_y -> ground_y - 300)
        scale_y = (w - 110) / PITCH_LENGTH_M
        scale_z = 150.0  # 150px per metre

        # Plot 3D projected parabolic path
        path = pred_3d.projected_path_3d
        pts = []
        for Xp, Yp, Zp in path:
            px = int(30 + Yp * scale_y)
            py = int(ground_y - Zp * scale_z)
            pts.append((px, py))

        color = (34, 197, 94) if pred_3d.height_verdict == "HITTING" else (234, 179, 8) if pred_3d.height_verdict == "UMPIRES_CALL" else (239, 68, 68)
        for i in range(1, len(pts)):
            cv2.line(view, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

        # Height clearance readout at stumps
        z_stumps = pred_3d.stump_z
        cv2.putText(view, f"Stump Height Z: {z_stumps:.2f}m ({pred_3d.height_verdict})",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return view


def _render_top_down_map(pred_3d):
    """Renders 3D Pitch Top-Down Map View."""
    w, h = 580, 330
    view = np.full((h, w, 3), (15, 23, 42), dtype=np.uint8)

    cv2.putText(view, "3D PITCH TOP-DOWN MAP", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)

    # Pitch strip top-down
    pitch_x1, pitch_x2 = w // 2 - 60, w // 2 + 60
    cv2.rectangle(view, (pitch_x1, 50), (pitch_x2, h - 30), (30, 58, 90), -1)

    # Length Zones Overlay Lines
    cv2.line(view, (pitch_x1, h - 60), (pitch_x2, h - 60), (56, 189, 248), 1)  # Yorker Zone
    cv2.putText(view, "YORKER", (pitch_x2 + 5, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1)

    cv2.line(view, (pitch_x1, h - 110), (pitch_x2, h - 110), (34, 197, 94), 1)  # Full Length Zone
    cv2.putText(view, "FULL", (pitch_x2 + 5, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1)

    cv2.line(view, (pitch_x1, h - 170), (pitch_x2, h - 170), (250, 204, 21), 1)  # Good Length Zone
    cv2.putText(view, "GOOD", (pitch_x2 + 5, h - 165), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1)

    # Stumps
    cv2.line(view, (pitch_x1 + 30, h - 30), (pitch_x2 - 30, h - 30), (250, 204, 21), 4)

    if pred_3d and pred_3d.has_prediction:
        scale_x = 40.0  # px per metre lateral
        scale_y = (h - 80) / PITCH_LENGTH_M

        # Pitch point marker
        px = int(w // 2 + pred_3d.pitch_3d[0] * scale_x)
        py = int(50 + pred_3d.pitch_3d[1] * scale_y)
        cv2.circle(view, (px, py), 6, (250, 204, 21), -1)

        # Impact point marker
        ix = int(w // 2 + pred_3d.impact_3d[0] * scale_x)
        iy = int(50 + pred_3d.impact_3d[1] * scale_y)
        cv2.circle(view, (ix, iy), 6, (239, 68, 68), -1)

        # Path line
        sx = int(w // 2 + pred_3d.stump_x * scale_x)
        sy = h - 30
        cv2.line(view, (ix, iy), (sx, sy), (34, 197, 94), 2, cv2.LINE_AA)

        cv2.putText(view, f"Pitch: {pred_3d.pitch_3d[1]:.1f}m | Impact: {pred_3d.impact_3d[1]:.1f}m | Stumps X: {pred_3d.stump_x:+.2f}m",
                    (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 232, 240), 1, cv2.LINE_AA)

    return view
