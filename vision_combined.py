#!/usr/bin/env python3
"""
vision_combined.py — QR-ONLY PHASE harness (debugged).
Active: camera init → undistort → QRPipeline (detect + template + match) → HUD/recording.
DISABLED THIS PHASE: planner, green banner, red zone, corridor nav, mission
logic, edge overlay (restore from git history when needed).
"""
import cv2
import numpy as np
from datetime import datetime
import os
import logging

import config
from main import get_undistort_maps
from qr_module import QRPipeline
from output_manager import OutputManager

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

# ── Camera: hardcoded to the TSTC USB camera (/dev/video0) ───────────────────
CAMERA_INDEX = 0   # from: v4l2-ctl --list-devices


def find_camera(width: int, height: int):
    print(f"[..] Opening camera index {CAMERA_INDEX} (TSTC USB20)...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at index {CAMERA_INDEX}. Unplugged?")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    # Drone-ready (uncomment for flight): manual short shutter freezes vibration blur
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 40)      # 40 = 4 ms ≈ 1/250 s (100µs units)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Index {CAMERA_INDEX} opened but cannot read frames.")

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] TSTC USB Camera opened at index {CAMERA_INDEX} ({aw}x{ah})")
    return cap, CAMERA_INDEX


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
    smoothing_window=getattr(config, "CONFIRM_WINDOW", 5),
    confirm_min_hits=getattr(config, "CONFIRM_MIN_HITS", 3),
    output_dir=out.run_dir,
    output_manager=out,
    clahe_limit=config.CLAHE_LIMIT,
    clahe_tile_size=config.CLAHE_TILE_SIZE)

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
        pipe.draw(frame, res)          # module draws the ONLY HUD

        # frame counter only (no duplicate HUD)
        cv2.putText(frame, f"FRAME: {frame_count}", (10, ACTUAL_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ── Logging ─────────────────────────────────────────────────────
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