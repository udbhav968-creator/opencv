"""
config.py
---------
Central place for every tunable constant used across the DRS project.
Change values here instead of hunting through the other files.
"""

# ---------------------------------------------------------------------------
# Frame / video settings
# ---------------------------------------------------------------------------
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FPS = 30

# ---------------------------------------------------------------------------
# Ball detection (HSV color range)
# Default is tuned for a RED cricket ball. If you use a white ball, switch
# to the WHITE_BALL_* range below by passing color_mode="white" to
# BallDetector.
# ---------------------------------------------------------------------------
RED_BALL_LOWER_1 = (0, 120, 70)
RED_BALL_UPPER_1 = (10, 255, 255)
RED_BALL_LOWER_2 = (170, 120, 70)
RED_BALL_UPPER_2 = (180, 255, 255)

WHITE_BALL_LOWER = (0, 0, 200)
WHITE_BALL_UPPER = (180, 40, 255)

MIN_BALL_RADIUS = 3
MAX_BALL_RADIUS = 25

# ---------------------------------------------------------------------------
# Pitch geometry (pixel coordinates in the processed frame)
# The "pitch" runs from the bowler's end (top of frame, far away / small)
# to the batsman's end (bottom of frame, close / large) -- a simple
# perspective approximation good enough for a single fixed camera demo.
# ---------------------------------------------------------------------------
BOWLER_END_Y = 60          # y-pixel representing the bowler's stumps line
BATSMAN_END_Y = 480        # y-pixel representing the batsman's stumps line (impact/crease line)

# Stump block width at each end (perspective: narrower far away, wider close)
STUMPS_WIDTH_FAR = 18
STUMPS_WIDTH_NEAR = 70
STUMPS_HEIGHT_FAR = 22
STUMPS_HEIGHT_NEAR = 90

FRAME_CENTER_X = FRAME_WIDTH // 2

# ---------------------------------------------------------------------------
# DRS zone colours (BGR, OpenCV order) -- match broadcast conventions
# ---------------------------------------------------------------------------
COLOR_RED = (0, 0, 255)          # Missing / not out zone marker
COLOR_GREEN = (0, 200, 0)        # Hitting / out zone marker
COLOR_YELLOW = (0, 220, 255)     # Umpire's call
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (20, 20, 20)
COLOR_STUMPS = (200, 200, 200)
COLOR_TRAJECTORY = (0, 165, 255)  # orange trail
COLOR_PREDICTED = (255, 0, 255)   # magenta dotted predicted path
COLOR_PITCH_MARKER = (0, 255, 255)

# ---------------------------------------------------------------------------
# Trajectory / physics
# ---------------------------------------------------------------------------
MIN_POINTS_FOR_FIT = 5      # minimum tracked points before we trust a polynomial fit
POLY_DEGREE_VERTICAL = 2    # y (down-track) vs depth -> parabolic bounce shape
POLY_DEGREE_LATERAL = 1     # x (sideways drift) vs depth -> near-linear swing/seam approx

# "Umpire's call" margin in pixels: if the predicted path is only marginally
# clipping the stumps, we flag it as umpire's call instead of a hard decision.
UMPIRES_CALL_MARGIN_PX = 6
