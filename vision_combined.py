#!/usr/bin/env python3
"""
vision_combined.py — QR-ONLY PHASE harness.
Active: camera init → undistort → QRPipeline (detect + template + match) → HUD/recording.
DISABLED THIS PHASE (commented): planner, green banner, red zone, corridor nav,
mission logic, edge overlay. Restore from git history when needed.
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

# ── [DISABLED] planner / other detectors / mission logic ─────────────────────
# from detectors.green_banner import GreenBannerDetector
# from detectors.red_zone      import RedZoneDetector
# from detectors.corridor_nav  import CorridorNavigator
# from mission_logic            import MissionController
# from planning.search_planner  import build_planner, draw_path, save_plan

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

print("=" * 60)
print("QR-ONLY PHASE: camera + QR detection + template matching")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = OutputManager(output_root=config.OUTPUT_DIR,
                    timestamp=timestamp,
                    candidate_throttle_sec=config.CANDIDATE_THROTTLE_SEC)


# ── Camera: STRICT index loop 4→0 (no silent internal fallback) ──────────────
# ── Camera: STRICT index loop 4→0 (no silent internal fallback) ──────────────
# ── Camera: Hardcoded to the TSTC USB Camera ──────────────────────────────────
def find_camera(width: int, height: int):
    """Forces OpenCV to use the TSTC USB camera at index 2."""
    REAL_INDEX = 2  
    
    print(f"[..] Forcing camera to index {REAL_INDEX} (TSTC USB20)...")
    cap = cv2.VideoCapture(REAL_INDEX, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at index {REAL_INDEX}. Is it unplugged?")
        
    # Configure camera settings
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    
    # Verify it can actually read frames
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Index {REAL_INDEX} opened but cannot read frames.")
        
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] TSTC USB Camera opened at index {REAL_INDEX} ({actual_w}x{actual_h})")
    return cap, REAL_INDEX


cap, cam_id = find_camera(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
ACTUAL_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ACTUAL_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
_map1, _map2, _, _ = get_undistort_maps(ACTUAL_W, ACTUAL_H)

video_path = os.path.join(out.run_dir, f"mission_{timestamp}.avi")
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'),
                         config.FPS, (ACTUAL_W, ACTUAL_H))

LOG_INTERVAL = 10
qr_log    = open(os.path.join(out.dir_logs, f"qr_{timestamp}.txt"),    "w", buffering=8192)
match_log = open(os.path.join(out.dir_logs, f"match_{timestamp}.txt"), "w", buffering=8192)

# ── The QR module (single facade) ────────────────────────────────────────────
pipe = QRPipeline(
    min_size=config.QR_MIN_SIZE,
    template_size=config.TEMPLATE_SIZE,
    grid_threshold=0.80,
    match_threshold=config.MATCH_THRESHOLD,
    sharpness_threshold=config.SHARPNESS_THRESHOLD,
    frame_size=(ACTUAL_W, ACTUAL_H),
    smoothing_window=3,
    output_dir=out.run_dir,
    output_manager=out,
    clahe_limit=config.CLAHE_LIMIT,
    clahe_tile_size=config.CLAHE_TILE_SIZE)

# ── [DISABLED] green/red/corridor/mission/planner construction ───────────────

frame_count = 0
try:
    while True:
        frame_count += 1
        now = datetime.now().strftime("%H:%M:%S.%f")

        ok, frame_bgr = cap.read()
        if not ok:
            logger.warning("[MAIN] cap.read() failed — skipping frame")
            continue
        frame_raw = cv2.remap(frame_bgr, _map1, _map2, cv2.INTER_LINEAR)
        frame = frame_raw.copy()

        # ── QR pipeline: detect → template load → match ──────────────────
        res = pipe.process_frame(frame_raw)
        pipe.draw(frame, res)

        # ── [DISABLED] edge overlay / green / red / corridor / mission / planner draws ──

        # ── Minimal HUD ──────────────────────────────────────────────────
        tmpl_text  = "LOADED" if res.template_loaded else "WAITING FOR START QR"
        tmpl_color = (0, 255, 0) if res.template_loaded else (0, 165, 255)
        cv2.putText(frame, f"TEMPLATE: {tmpl_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tmpl_color, 2)
        cv2.putText(frame, f"QR: {len(res.detections)}  CAND: {len(res.candidates)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if res.best is not None:
            cc = (0, 255, 0) if res.best.matched else (0, 0, 255)
            cv2.putText(frame,
                        f"MATCH: {'YES' if res.best.matched else 'NO'}  "
                        f"G:{res.best.grid_conf:.2f} M:{res.best.matcher_conf:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cc, 2)
        if res.confirmed:
            cv2.putText(frame, f"CONFIRMED conf={res.confirmed_conf:.2f}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FRAME: {frame_count}", (10, ACTUAL_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ── Logging ──────────────────────────────────────────────────────
        if frame_count % LOG_INTERVAL == 0:
            if res.detections:
                for ds, (cxq, cyq), rect, raw_crop, _ in res.detections:
                    has_img = 1 if raw_crop is not None else 0
                    ds_short = (ds[:40] + "…") if len(ds) > 40 else ds
                    qr_log.write(f"{now},QR,1,{cxq},{cyq},{ds_short},{has_img}\n")
            else:
                qr_log.write(f"{now},QR,0,0,0,None,0\n")

            if res.best is not None:
                match_log.write(
                    f"{now},{res.best.matched},{res.best.grid_conf:.4f},"
                    f"{res.best.matcher_conf:.4f},{res.best.center},"
                    f"confirmed={res.confirmed}\n")

            logger.info(f"[FRAME {frame_count}] QR={len(res.detections)} "
                        f"cand={len(res.candidates)} "
                        f"conf={res.confirmed_conf:.2f} confirmed={res.confirmed}")

        writer.write(frame)
        cv2.imshow("QR-ONLY Mission Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
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