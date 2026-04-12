# vision_combined.py
#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
from datetime import datetime
import os
import config
from qr_detector import QRDetector
from green_banner import GreenBannerDetector
from red_zone import RedZoneDetector
from corridor_nav import CorridorNavigator
from mission_logic import MissionController

print("="*60)
print("OPTIMIZED MISSION VISION + LOGIC")
print("="*60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
log_dir = f"{config.OUTPUT_DIR}/logs_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": config.RESOLUTION, "format": "RGB888"}
))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AwbEnable": True})

# Video
video_path = f"{config.OUTPUT_DIR}/mission_{timestamp}.avi"
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 
                        config.FPS, config.RESOLUTION)

# QR Log only
qr_log = open(f"{log_dir}/qr_{timestamp}.txt", "w")

# Detectors
qr = QRDetector(config.QR_MIN_SIZE)
green = GreenBannerDetector(config.GREEN_HSV[0], config.GREEN_HSV[1], config.MIN_AREA)
red = RedZoneDetector(config.RED_HSV[0][0], config.RED_HSV[0][1],
                     config.RED_HSV[1][0], config.RED_HSV[1][1], config.MIN_AREA)
nav = CorridorNavigator(config.CORRIDOR_TOLERANCE, config.CORRIDOR_TOLERANCE)

# Mission
mission = MissionController(target_qr="DROP_ZONE_A")

cx, cy = config.RESOLUTION[0]//2, config.RESOLUTION[1]//2
frame_count = 0

try:
    while True:
        frame_count += 1
        now = datetime.now().strftime("%H:%M:%S.%f")
        
        frame = picam2.capture_array()
        
        # Detection
        qr_results = qr.detect(frame)
        green_results = green.detect(frame)
        red_results = red.detect(frame)
        nav_output = nav.detect(frame)
        
        center_line, deviation = nav_output if isinstance(nav_output, tuple) else (None, 0)
        
        # Mission logic
        command = mission.process(qr_results, green_results, red_results, deviation)
        debug_info = mission.get_debug_info()
        
        # Visual feedback
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[:, :, 1:] = 0
        frame = cv2.addWeighted(frame, 1.0, edges_colored, 0.6, 0)
        
        qr.draw(frame, qr_results)
        green.draw(frame, green_results)
        red.draw(frame, red_results)
        nav.draw(frame, center_line, deviation)
        
        if center_line is not None:
            cv2.circle(frame, (center_line, cy), 6, (0, 0, 255), -1)
            cv2.line(frame, (cx, cy), (center_line, cy), (255, 0, 255), 2)
        
        direction = "Centered" if abs(deviation) < 20 else ("MOVE RIGHT" if deviation > 0 else "MOVE LEFT")
        
        cv2.putText(frame, f"STATE: {debug_info['state']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)
        cv2.putText(frame, f"CMD: {command}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_CENTER, 2)
        cv2.putText(frame, direction, (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_CENTER, 2)
        
        cv2.line(frame, (cx, 0), (cx, config.RESOLUTION[1]), config.COLOR_CENTER, 1)
        
        # QR logging only
        if qr_results:
            for data, center in qr_results:
                cx_qr, cy_qr = center
                qr_log.write(f"{now},QR,1,{cx_qr},{cy_qr},{data}\n")
        else:
            qr_log.write(f"{now},QR,0,0,0,None\n")
        
        # Terminal output
        if frame_count % 10 == 0:
            print(f"[{debug_info['state']}] {command} | Timer: {debug_info['timer']}")
        
        writer.write(frame)
        cv2.imshow("Mission Vision", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("Error:", e)

finally:
    picam2.stop()
    writer.release()
    qr_log.close()
    cv2.destroyAllWindows()
    print("✓ Video:", video_path)
    print("✓ QR Log:", log_dir)