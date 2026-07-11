"""
stump_zone.py
-------------
Defines the stumps' geometry (with simple perspective interpolation
between the bowler's end and batsman's end) and classifies trajectory
points into DRS-style zones:

  - Pitching:  Outside Leg / In-line / Outside Off
  - Impact:    Outside Leg / In-line / Outside Off
  - Wickets:   Hitting / Umpire's Call / Missing

This is a simplified single-axis (lateral only) model -- a real system
also reasons about height (would the ball have gone over the stumps).
We call that out explicitly in the README as a known limitation.
"""

import config as cfg


def stumps_half_width_at(y_depth):
    """
    Linearly interpolate the stumps' half-width (in pixels) at a given
    y-depth, based on the configured near/far widths (simple perspective
    approximation for a fixed camera).
    """
    y0, y1 = cfg.BOWLER_END_Y, cfg.BATSMAN_END_Y
    w0, w1 = cfg.STUMPS_WIDTH_FAR, cfg.STUMPS_WIDTH_NEAR

    t = (y_depth - y0) / float(y1 - y0) if y1 != y0 else 0.0
    t = max(0.0, min(1.0, t))
    width = w0 + t * (w1 - w0)
    return width / 2.0


def stump_center_x_at(y_depth):
    """Stumps are assumed centered on the frame's vertical midline."""
    return cfg.FRAME_CENTER_X


def classify_lateral_zone(x, y_depth):
    """
    Classifies a lateral position as relative to the stumps at that
    depth: returns one of "OUTSIDE_LEG", "IN_LINE", "OUTSIDE_OFF".

    NOTE: "off side" vs "leg side" is batsman-handedness dependent in
    real cricket. For this simplified demo we treat +x (right of centre
    in the frame) as "off" and -x as "leg" -- flip STUMPS lookups in your
    own footage if your batsman is left-handed / camera is reversed.
    """
    half_width = stumps_half_width_at(y_depth)
    center = stump_center_x_at(y_depth)
    offset = x - center

    if offset < -half_width:
        return "OUTSIDE_LEG"
    elif offset > half_width:
        return "OUTSIDE_OFF"
    else:
        return "IN_LINE"


def classify_wicket_hit(predicted_x, y_depth=cfg.BATSMAN_END_Y):
    """
    Classifies whether the predicted trajectory hits the stumps, is an
    umpire's call (marginal), or misses.
    Returns one of "HITTING", "UMPIRES_CALL", "MISSING".
    """
    half_width = stumps_half_width_at(y_depth)
    center = stump_center_x_at(y_depth)
    offset = abs(predicted_x - center)

    if offset <= half_width - cfg.UMPIRES_CALL_MARGIN_PX:
        return "HITTING"
    elif offset <= half_width + cfg.UMPIRES_CALL_MARGIN_PX:
        return "UMPIRES_CALL"
    else:
        return "MISSING"


def get_stump_box(y_depth):
    """
    Returns (x1, y1, x2, y2) pixel box approximating the stumps at a
    given depth, for drawing.
    """
    half_width = stumps_half_width_at(y_depth)
    center = stump_center_x_at(y_depth)

    y0, y1 = cfg.BOWLER_END_Y, cfg.BATSMAN_END_Y
    h0, h1 = cfg.STUMPS_HEIGHT_FAR, cfg.STUMPS_HEIGHT_NEAR
    t = (y_depth - y0) / float(y1 - y0) if y1 != y0 else 0.0
    t = max(0.0, min(1.0, t))
    height = h0 + t * (h1 - h0)

    x1 = int(center - half_width)
    x2 = int(center + half_width)
    y2 = int(y_depth)
    y1_box = int(y_depth - height)
    return x1, y1_box, x2, y2
