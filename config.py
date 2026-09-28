# config.py — Drone Mission Configuration
# USB camera fork: SINOSEE global-shutter module (replaces Picam3)

# ── Hardware / Camera (USB — see main.initialize_camera) ──────
CAMERA_ID       = 1          # V4L2 index of the USB module (/dev/videoN)
CAMERA_WIDTH    = 1600
CAMERA_HEIGHT   = 1300
OUTPUT_DIR      = "./output"
RESOLUTION      = (CAMERA_WIDTH, CAMERA_HEIGHT)   # used by writer/HUD/matcher
FPS             = 30

# ── Color detection HSV ranges ────────────────────────────────
GREEN_HSV_LOWER = (40, 40, 40)
GREEN_HSV_UPPER = (80, 255, 255)
GREEN_HSV = (GREEN_HSV_LOWER, GREEN_HSV_UPPER)
GREEN_MIN_AREA = 500

RED_HSV_LOWER1 = (0,   100, 100)
RED_HSV_UPPER1 = (10,  255, 255)
RED_HSV_LOWER2 = (170, 100, 100)
RED_HSV_UPPER2 = (180, 255, 255)
RED_HSV = ((RED_HSV_LOWER1, RED_HSV_UPPER1),
           (RED_HSV_LOWER2, RED_HSV_UPPER2))

# ── QR / Detection ────────────────────────────────────────────
QR_MIN_SIZE = 50
MIN_AREA    = 500

# ── Navigation ────────────────────────────────────────────────
CORRIDOR_EDGE_THRESHOLD  = 50     # used by detectors/corridor_nav.py
CORRIDOR_TOLERANCE       = 20

# ── Red zone failsafe ─────────────────────────────────────────
RED_DANGER_AREA      = 5000   # used by mission_logic.py override
MIN_RED_AREA         = 5000   # used by detectors/red_zone.py

# ── Template matcher ─────────────────────────────────────────
MATCH_THRESHOLD  = 0.75
SHARPNESS_THRESHOLD = 25.0     # Laplacian variance below this = blurry candidate
TEMPLATE_SIZE    = (128, 128)
CLAHE_LIMIT      = 2.0
CLAHE_TILE_SIZE  = (8, 8)

# ── Search planning ───────────────────────────────────────────
# All planning waypoints are in NORMALISED coordinates (0.0 → 1.0).
# (0,0) = bottom-left of plot, (1,1) = top-right.
PLAN_ENABLED           = True       # set False to disable planner entirely
PLAN_STRATEGY          = "boustrophedon"
PLAN_ROWS              = 8          # number of sweep rows
PLAN_DIRECTION         = "bottom_left_to_bottom_right"
PLAN_EARLY_EXIT_ON_QR  = True       # stop walking the path once template matched
PLAN_VISUAL_MARGIN_PX  = 30         # padding inside frame when drawing path
PLAN_WAYPOINT_HOLD_SEC = 1.0        # visualisation: time spent at each waypoint
PLAN_COLOR_PATH        = (0, 200, 255)
PLAN_COLOR_VISITED     = (80,  80, 200)
PLAN_COLOR_CURRENT     = (0, 255,   0)
PLAN_COLOR_PENDING     = (180, 180, 180)

# ── Output manager (NEW) ──────────────────────────────────────
SAVE_FIRST_QR     = True
SAVE_TEMPLATE_QR  = True
SAVE_CANDIDATES   = True
CANDIDATE_THROTTLE_SEC = 1.0        # min seconds between candidate saves

# ── Visual debug colors (BGR) ─────────────────────────────────
COLOR_QR           = (255,   0, 255)
COLOR_GREEN        = (  0, 255,   0)
COLOR_RED          = (  0,   0, 255)
COLOR_CORRIDOR     = (255, 255,   0)
COLOR_CENTER       = (  0, 255, 255)
COLOR_TEXT         = (255, 255, 255)
COLOR_FRAME_CENTER = (  0, 255, 255)
COLOR_BBOX         = (255, 165,   0)
COLOR_OBJ_CENTER   = (  0, 255,   0)

CONFIRM_WINDOW        = 5       # N frames in confirmation window
CONFIRM_MIN_HITS      = 3       # M hits required → vibration tolerant