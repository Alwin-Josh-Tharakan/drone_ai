# config.py — Drone Mission Configuration
# Raspberry Pi 4B optimized

# ── Hardware ──────────────────────────────────────────────────
YOLO_MODEL = "/home/pigec/yolo/yolov8n.pt"
OUTPUT_DIR = "/home/pigec/drone_mission/output"
RESOLUTION = (640, 480)
FPS = 30
CONFIDENCE_THRESHOLD = 0.5

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
CORRIDOR_EDGE_THRESHOLD  = 50
CORRIDOR_CENTER_TOLERANCE = 20
CORRIDOR_TOLERANCE = 20
GREEN_TOLERANCE    = 30
QR_TOLERANCE       = 10
STABLE_FRAMES      = 90

# ── Red zone failsafe ─────────────────────────────────────────
RED_DANGER_AREA      = 5000
MIN_RED_AREA         = 5000
MIN_RED_SOLIDITY     = 0.70
MIN_RED_ASPECT_RATIO = 0.40
MAX_ESCAPE_TIME      = 5.0

# ── Template matcher ─────────────────────────────────────────
MATCH_THRESHOLD  = 0.75
TEMPLATE_SIZE    = (128, 128)
CLAHE_LIMIT      = 2.0
CLAHE_TILE_SIZE  = (8, 8)

# ── Search grid (legacy) ──────────────────────────────────────
GRID_SIZE = (10, 10)

# ── Search planning (NEW) ─────────────────────────────────────
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
CANDIDATE_THROTTLE_SEC = 2.0        # min seconds between candidate saves

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
COLOR_OFFSET_LINE  = (255,   0,   0)