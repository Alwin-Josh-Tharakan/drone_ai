# config.py
YOLO_MODEL = "/home/pigec/yolo/yolov8n.pt"
OUTPUT_DIR = "/home/pigec/drone_mission/output"
RESOLUTION = (640, 480)
FPS = 30
CONFIDENCE_THRESHOLD = 0.5

# Color detection HSV ranges
GREEN_HSV = ((40, 40, 40), (80, 255, 255))
RED_HSV = (((0, 100, 100), (10, 255, 255)), ((170, 100, 100), (180, 255, 255)))
MIN_AREA = 500
QR_MIN_SIZE = 50

# Navigation
CORRIDOR_TOLERANCE = 20
GREEN_TOLERANCE = 30
QR_TOLERANCE = 10
STABLE_FRAMES = 90
RED_DANGER_AREA = 5000

# Visual debug colors (BGR)
COLOR_QR = (255, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_CORRIDOR = (255, 255, 0)
COLOR_CENTER = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)