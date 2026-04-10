#!/usr/bin/env python3
"""Phase 2: Debugged Version (Stable + Clean)"""

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
from datetime import datetime
import os
import config

from detectors.qr_detector import QRDetector
from detectors.green_banner import GreenBannerDetector
from detectors.red_zone import RedZoneDetector
from detectors.corridor_nav import CorridorNavigator

# { INITIALIZATION START }
print("="*60)
print("PHASE 2: DEBUGGED MISSION VISION")
print("="*60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
log_dir = f"{config.OUTPUT_DIR}/logs_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# CAMERA
picam2 = Picamera2()
cam_config = picam2.create_preview_configuration(
    main={"size": config.RESOLUTION, "format": "RGB888"}
)
picam2.configure(cam_config)
picam2.start()

picam2.set_controls({
    "AfMode": controls.AfModeEnum.Continuous,
    "AwbEnable": True
})

# VIDEO
video_path = f"{config.OUTPUT_DIR}/mission_{timestamp}.avi"
writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    config.FPS,
    config.RESOLUTION
)

# LOGS
qr_log = open(f"{log_dir}/qr_{timestamp}.txt", "w")
green_log = open(f"{log_dir}/green_{timestamp}.txt", "w")
red_log = open(f"{log_dir}/red_{timestamp}.txt", "w")
nav_log = open(f"{log_dir}/nav_{timestamp}.txt", "w")

# DETECTORS
qr = QRDetector(config.QR_MIN_SIZE)
green = GreenBannerDetector(config.GREEN_HSV_LOWER, config.GREEN_HSV_UPPER, config.GREEN_MIN_AREA)
red = RedZoneDetector(
    config.RED_HSV_LOWER1, config.RED_HSV_UPPER1,
    config.RED_HSV_LOWER2, config.RED_HSV_UPPER2,
    config.RED_MIN_AREA
)
nav = CorridorNavigator(config.CORRIDOR_EDGE_THRESHOLD, config.CORRIDOR_CENTER_TOLERANCE)

cx = config.RESOLUTION[0] // 2
cy = config.RESOLUTION[1] // 2
frame_count = 0
# { INITIALIZATION END }

try:
    while True: # { MAIN_LOOP_START }
        frame_count += 1
        now = datetime.now().strftime("%H:%M:%S.%f")

        # RAW FRAME (for detection)
        frame_raw = picam2.capture_array()

        # DISPLAY FRAME (copy)
        frame = frame_raw.copy()

        # ---------------- DETECTION ----------------
        qr_results = qr.detect(frame_raw)
        green_results = green.detect(frame_raw)
        red_results = red.detect(frame_raw)

        nav_output = nav.detect(frame_raw)
        if isinstance(nav_output, tuple):
            center_line = nav_output[0]
            deviation = nav_output[1] if len(nav_output) > 1 else 0
        else:
            center_line = None
            deviation = 0

        # ---------------- EDGE DEBUG (VISUAL ONLY) ----------------
        gray = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[:, :, 1:] = 0
        frame = cv2.addWeighted(frame, 1.0, edges_colored, 0.6, 0)

        # ---------------- DRAW ----------------
        qr.draw(frame, qr_results)
        green.draw(frame, green_results)
        red.draw(frame, red_results)
        nav.draw(frame, center_line, deviation)
        
        # Draw corridor center point
        if center_line is not None:
            cv2.circle(frame, (center_line, cy), 6, (0, 0, 255), -1)
            
            # Draw deviation line 
            cv2.line(frame, (cx, cy), (center_line, cy), (255, 0, 255), 2)
            
        # Direction text
        if abs(deviation) < 20:
            direction = "Centered"
        elif deviation > 0:
            direction = "MOVE RIGHT"
        else:
            direction = "MOVE LEFT"
            
        cv2.putText(frame, direction, (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Center line
        cv2.line(frame, (cx, 0), (cx, config.RESOLUTION[1]), (0, 255, 255), 1)

        # ---------------- LOGGING ----------------
        # QR
        if qr_results: # { QR_LOG_START }
            for result in qr_results:
                data, center = result
                cx_qr, cy_qr = center
                qr_log.write(f"{now},QR,1,{cx_qr},{cy_qr},0,0,{data}\n")
        else:
            qr_log.write(f"{now},QR,0,0,0,0,0,None\n")
        # { QR_LOG_END }

        # GREEN
        for (cx_g, cy_g), area in green_results: # { GREEN_LOG_START }
            dx, dy = cx_g - cx, cy_g - cy
            green_log.write(f"{now},GREEN,1,{cx_g},{cy_g},{dx},{dy},{area}\n")
        # { GREEN_LOG_END }

        # RED
        for (cx_r, cy_r), area in red_results: # { RED_LOG_START }
            dx, dy = cx_r - cx, cy_r - cy
            red_log.write(f"{now},RED,1,{cx_r},{cy_r},{dx},{dy},{area}\n")
        # { RED_LOG_END }

        # NAV
        nav_log.write(f"{now},NAV,1,0,0,{deviation},0,0\n")

        # ---------------- OUTPUT ----------------
        writer.write(frame)
        cv2.imshow("Mission Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # { MAIN_LOOP_END }

except Exception as e:
    print("Error:", e)

finally: # { CLEANUP_START }
    picam2.stop()
    writer.release()
    qr_log.close()
    green_log.close()
    red_log.close()
    nav_log.close()
    cv2.destroyAllWindows()

    print("✓ Video:", video_path)
    print("✓ Logs:", log_dir)
    print("✓ Shutdown Complete")
# { CLEANUP_END }
