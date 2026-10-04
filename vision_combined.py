#!/usr/bin/env python3
"""
vision_combined.py — HIGH-RES GSTREAMER RECEIVER + QR DETECTION.
Receives 1600x1200 MJPEG stream from Pi, processes QR codes, and displays 
full-resolution overlay in a maximized window.
"""
import cv2
import numpy as np
from datetime import datetime
import os
import logging

import config
from main import initialize_camera, get_undistort_maps
from qr_module import QRPipeline
from output_manager import OutputManager

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

print("=" * 60)
print("HIGH-RES GSTREAMER RECEIVER: 1600x1200 + QR DETECTION")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OutputManager(output_root=config.OUTPUT_DIR,
                    timestamp=timestamp,
                    candidate_throttle_sec=config.CANDIDATE_THROTTLE_SEC)

# ── Initialize High-Res GStreamer Receiver ────────────────────────────────────
# We force 1600x1200 to match the Pi's transmission
cap = initialize_camera(gst_port=5000)

ACTUAL_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ACTUAL_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if ACTUAL_W != 1600 or ACTUAL_H != 1200:
    logger.warning(f"[MAIN] Expected 1600x1200 but got {ACTUAL_W}x{ACTUAL_H}. Check Pi command.")

print(f"[MAIN] Receiving Stream: {ACTUAL_W}x{ACTUAL_H}")

# Precompute Undistortion Maps for 1600x1200
_map1, _map2, _, _ = get_undistort_maps(ACTUAL_W, ACTUAL_H)

# ── Video Writer (Optional: Record the processed output) ──────────────────────
video_path = os.path.join(out.run_dir, f"mission_{timestamp}.avi")
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'),
                         config.FPS, (ACTUAL_W, ACTUAL_H))

LOG_INTERVAL = 10
qr_log    = open(os.path.join(out.dir_logs, f"qr_{timestamp}.txt"),    "w", buffering=8192)
match_log = open(os.path.join(out.dir_logs, f"match_{timestamp}.txt"), "w", buffering=8192)

# ── The QR Module (Single Facade) ─────────────────────────────────────────────
pipe = QRPipeline(
    min_size=config.QR_MIN_SIZE,       # e.g., 50px minimum for detection
    template_size=config.TEMPLATE_SIZE,# e.g., 128x128 binary template
    grid_threshold=0.80,               # Grid match threshold
    match_threshold=config.MATCH_THRESHOLD, # Final visual match threshold
    sharpness_threshold=config.SHARPNESS_THRESHOLD, # Blur guard
    frame_size=(ACTUAL_W, ACTUAL_H),   # IMPORTANT: Pass actual res here
    smoothing_window=getattr(config, "CONFIRM_WINDOW", 5),
    confirm_min_hits=getattr(config, "CONFIRM_MIN_HITS", 3),
    output_dir=out.run_dir,
    output_manager=out,
    clahe_limit=config.CLAHE_LIMIT,
    clahe_tile_size=config.CLAHE_TILE_SIZE)

# ── Display Configuration ─────────────────────────────────────────────────────
WINDOW_NAME = "Mission Vision (Processed)"

# Create window that CAN be resized/maximized
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Attempt to set initial size to match typical laptop screens for visibility
# If you have a 1920x1080 screen, this fills it nicely while keeping aspect ratio
SCREEN_W = 1920
SCREEN_H = 1080
cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)

frame_count = 0
last_res = None

try:
    while True:
        frame_count += 1
        now = datetime.now().strftime("%H:%M:%S.%f")

        # 1. Read Raw Frame from GStreamer
        ok, frame_bgr = cap.read()
        if not ok:
            logger.warning("[MAIN] cap.read() failed — skipping frame")
            continue
            
        # 2. Undistort (Lens Correction)
        frame_raw = cv2.remap(frame_bgr, _map1, _map2, cv2.INTER_LINEAR)
        
        # 3. Copy for Overlay Drawing (Don't draw on the raw input directly if possible)
        frame_display = frame_raw.copy()

        # 4. RUN QR DETECTION PIPELINE
        # This performs: Detect -> Template Load -> Visual Match -> Temporal Smooth
        res = pipe.process_frame(frame_raw)
        last_res = res
        
        # 5. Draw Overlays (Boxes, HUD, Confidence Bars)
        # Note: pipe.draw modifies 'frame_display' in place
        pipe.draw(frame_display, res)

        # 6. Add Frame Counter Text
        cv2.putText(frame_display, f"FRAME: {frame_count}", 
                    (10, ACTUAL_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 7. Log Data (Every N frames)
        if frame_count % LOG_INTERVAL == 0 and last_res is not None:
            if last_res.detections:
                for ds, (cxq, cyq), rect, raw_crop, _ in last_res.detections:
                    has_img = 1 if raw_crop is not None else 0
                    ds_short = (ds[:40] + "…") if len(ds) > 40 else ds
                    qr_log.write(f"{now},QR,1,{cxq},{cyq},{ds_short},{has_img}\n")
            else:
                qr_log.write(f"{now},QR,0,0,0,None,0\n")

            if last_res.best is not None:
                match_log.write(
                    f"{now},{last_res.best.matched},"
                    f"{last_res.best.grid_conf:.4f},"
                    f"{last_res.best.matcher_conf:.4f},"
                    f"{last_res.best.center},"
                    f"confirmed={last_res.confirmed}\n")

            logger.info(f"[FRAME {frame_count}] QR={len(last_res.detections)} "
                        f"cand={len(last_res.candidates)} "
                        f"conf={last_res.confirmed_conf:.2f} confirmed={last_res.confirmed}")

        # 8. Write Video (Optional - Uncomment to save processed video)
        # writer.write(frame_display)

        # 9. SHOW OUTPUT WINDOW
        cv2.imshow(WINDOW_NAME, frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    logger.info("Manual termination requested (Ctrl+C).")
except Exception as e:
    logger.exception(f"Main loop error: {e}")

finally:
    cap.release()
    writer.release()
    for f in (qr_log, match_log):
        try:
            f.close()
        except Exception:
            pass
    cv2.destroyAllWindows()
    logger.info(f"✓ Run dir : {out.run_dir}")
    logger.info(f"✓ Video   : {video_path}")
    logger.info("✓ Shutdown complete")