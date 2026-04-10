"""Configuration for drone vision pipeline"""

# Absolute paths
YOLO_MODEL = "/home/pigec/yolo/yolov8n.pt"
OUTPUT_DIR = "/home/pigec/drone_mission/output"
CALIBRATION_FILE = "/home/pigec/drone/calibration_data.npz"  # For Phase 2+

# Camera settings
RESOLUTION = (640, 480)
FPS = 30

# Detection settings
CONFIDENCE_THRESHOLD = 0.5

# Display colors (BGR)
COLOR_FRAME_CENTER = (255, 255, 0)  # Cyan
COLOR_BBOX = (0, 255, 0)            # Green
COLOR_OBJ_CENTER = (0, 0, 255)      # Red
COLOR_OFFSET_LINE = (0, 255, 255)   # Yellow
COLOR_TEXT = (0, 255, 0)            # Green

# ... existing config ...

# QR Detection
QR_MIN_SIZE = 50  # pixels

# Color detection HSV ranges
GREEN_HSV_LOWER = (40, 40, 40)
GREEN_HSV_UPPER = (80, 255, 255)
GREEN_MIN_AREA = 500

RED_HSV_LOWER1 = (0, 100, 100)    # Red wraps around hue
RED_HSV_UPPER1 = (10, 255, 255)
RED_HSV_LOWER2 = (170, 100, 100)
RED_HSV_UPPER2 = (180, 255, 255)
RED_MIN_AREA = 500

# Corridor navigation
CORRIDOR_EDGE_THRESHOLD = 50
CORRIDOR_CENTER_TOLERANCE = 30
