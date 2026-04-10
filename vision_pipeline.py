#!/usr/bin/env python3
"""
Phase 1: Live YOLO Detection with Center Offset Tracking
"""

import cv2
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
from ultralytics import YOLO
import os
import config

# Create output directory
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 1: LIVE YOLO DETECTION + CENTER OFFSET")
print("="*60)

# Initialize camera
print("[1/4] Initializing Picamera2...")
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": config.RESOLUTION, "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()
from libcamera import controls
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})  # Continuous autofocus
print("✓ Camera ready")

# Load YOLO
print("[2/4] Loading YOLO model...")
model = YOLO(config.YOLO_MODEL, task='detect')
labels = model.names
print(f"✓ Model loaded: {len(labels)} classes")

# Setup video writer
print("[3/4] Setting up video recorder...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{config.OUTPUT_DIR}/detect_{timestamp}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(output_path, fourcc, config.FPS, config.RESOLUTION)
print(f"✓ Recording to: {output_path}")

# Frame center
cx = config.RESOLUTION[0] // 2
cy = config.RESOLUTION[1] // 2

print("[4/4] Starting detection loop...")
print("Press 'q' to quit\n" + "-"*60)

frame_count = 0

try:
    while True:
        frame_count += 1

        # Capture
        frame = picam2.capture_array()
        
        # YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes
        
        # Draw frame center crosshair
        cv2.circle(frame, (cx, cy), 5, config.COLOR_FRAME_CENTER, -1)
        cv2.line(frame, (cx-15, cy), (cx+15, cy), config.COLOR_FRAME_CENTER, 2)
        cv2.line(frame, (cx, cy-15), (cx, cy+15), config.COLOR_FRAME_CENTER, 2)
        
        # Find best target (largest area)
        best_det = None
        best_area = 0

        for i in range(len(detections)):
            conf = detections[i].conf.item()
            if conf < config.CONFIDENCE_THRESHOLD:
                continue

            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            area = (xmax - xmin) * (ymax - ymin)

            if area > best_area:
                best_area = area
                best_det = detections[i]

        # Process only best target
        if best_det is not None:
            xyxy = best_det.xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(best_det.cls.item())
            classname = labels[classidx]
            conf = best_det.conf.item()

            ox = (xmin + xmax) // 2
            oy = (ymin + ymax) // 2

            dx = ox - cx
            dy = oy - cy

            h_dir = "CENTERED" if abs(dx) < 20 else ("RIGHT" if dx > 0 else "LEFT")
            v_dir = "CENTERED" if abs(dy) < 20 else ("DOWN" if dy > 0 else "UP")

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), config.COLOR_BBOX, 2)
            cv2.circle(frame, (ox, oy), 5, config.COLOR_OBJ_CENTER, -1)
            cv2.line(frame, (cx, cy), (ox, oy), config.COLOR_OFFSET_LINE, 2)

            label = f'{classname}: {int(conf*100)}%'
            cv2.putText(frame, label, (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2)

            cv2.putText(frame, f"dx:{dx} dy:{dy}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)
            cv2.putText(frame, f"{h_dir} | {v_dir}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)

            if frame_count % 10 == 0:
                print(f"[TARGET] {classname:12s} | conf:{conf:.2f} | "
                      f"offset:({dx:+4d},{dy:+4d}) | {h_dir:8s} {v_dir:8s}")
        
        # Display
        cv2.imshow('Phase 1: YOLO Detection + Offset', frame)
        
        # Record
        writer.write(frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[!] Interrupted")
    
finally:
    print("\n" + "="*60)
    print("SHUTDOWN")
    print("="*60)
    picam2.stop()
    writer.release()
    cv2.destroyAllWindows()
    print(f"✓ Video saved: {output_path}")
    print("✓ Done")
