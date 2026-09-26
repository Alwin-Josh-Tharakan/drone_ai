#!/usr/bin/env python3
"""
vision_combined.py
FULL RUN: QR detection + template matching + mission logic + navigation
          + Boustrophedon search planning + dataset capture.

WORKFLOW
--------
1. STARTUP
   * Build OutputManager → makes run_<ts>/ tree
   * Build planner (Boustrophedon, normalised 0–1 coords).  ANY planner
     failure silently falls back to NullPlanner — main loop never crashes.

2. PER FRAME
   * Capture frame → run detectors (QR, green, red, corridor).
   * On FIRST QR detection: decode payload → build template → save copies
     into first_qr/ + template_qr/.
   * For every non-START QR: sharpness check, visual match, throttled save
     into candidate_qr/.
   * Temporal smoothing for HUD match status.
   * Run MissionController (which uses QRDetector.match_against_target).
   * Advance planner waypoint (only if early-exit not triggered).
   * Draw all detector overlays + planned path + HUD.

3. SHUTDOWN
   * Save waypoints + path_preview.png.
   * Close all log files.
"""

import cv2
import numpy as np
from datetime import datetime
import os
import logging

import config
from main                    import initialize_camera, get_undistort_maps
from detectors.qr_detector       import QRDetector, TargetData
from detectors.template_matcher  import TemplateMatcher, MatchResult
from detectors.green_banner      import GreenBannerDetector
from detectors.red_zone          import RedZoneDetector
from detectors.corridor_nav      import CorridorNavigator
from mission_logic               import MissionController
from output_manager              import OutputManager

# ── Crash-safe planner import ─────────────────────────────────────────────────
PLANNER_AVAILABLE = True
try:
    from planning.search_planner import (
        build_planner, draw_path, save_plan)
except Exception as _e:
    PLANNER_AVAILABLE = False

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

if not PLANNER_AVAILABLE:
    logger.warning(
        "[MAIN] planning.search_planner failed to import — "
        "planner will be disabled this run.")

# ── Banner ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("OPTIMIZED MISSION VISION + LOGIC + TEMPLATE MATCHING + PLANNER")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Output manager (creates per-run tree) ─────────────────────────────────────
out = OutputManager(
    output_root            = config.OUTPUT_DIR,
    timestamp              = timestamp,
    candidate_throttle_sec = config.CANDIDATE_THROTTLE_SEC)

# ── Camera (USB SINOSEE global-shutter module) ───────────────────────────────
cap = initialize_camera(camera_id=config.CAMERA_ID,
                        width=config.CAMERA_WIDTH,
                        height=config.CAMERA_HEIGHT)
ACTUAL_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ACTUAL_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Undistortion maps for the ACTUAL negotiated resolution (cached — remap is
# ~10x faster than per-frame cv2.undistort)
_map1, _map2 = get_undistort_maps(ACTUAL_W, ACTUAL_H)

# ── Video writer ──────────────────────────────────────────────────────────────
video_path = os.path.join(out.run_dir, f"mission_{timestamp}.avi")
writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    config.FPS,
    config.RESOLUTION)

# ── Logs (placed inside run_<ts>/logs/) ───────────────────────────────────────
LOG_INTERVAL = 10
qr_log      = open(os.path.join(out.dir_logs, f"qr_{timestamp}.txt"),      "w", buffering=8192)
match_log   = open(os.path.join(out.dir_logs, f"match_{timestamp}.txt"),   "w", buffering=8192)
green_log   = open(os.path.join(out.dir_logs, f"green_{timestamp}.txt"),   "w", buffering=8192)
red_log     = open(os.path.join(out.dir_logs, f"red_{timestamp}.txt"),     "w", buffering=8192)
nav_log     = open(os.path.join(out.dir_logs, f"nav_{timestamp}.txt"),     "w", buffering=8192)
mission_log = open(os.path.join(out.dir_logs, f"mission_{timestamp}.txt"), "w", buffering=8192)
plan_log    = open(os.path.join(out.dir_logs, f"plan_{timestamp}.txt"),    "w", buffering=8192)

# ── Detectors ─────────────────────────────────────────────────────────────────
qr = QRDetector(
    config.QR_MIN_SIZE,
    output_dir    = out.run_dir,        # so QRDetector internal saves
    template_size = config.TEMPLATE_SIZE,
    sharpness_min = 100.0)

matcher = TemplateMatcher(
    match_threshold = config.MATCH_THRESHOLD,
    template_size   = config.TEMPLATE_SIZE,
    clahe_limit     = config.CLAHE_LIMIT,
    clahe_tile_size = config.CLAHE_TILE_SIZE,
    frame_size      = config.RESOLUTION)

green = GreenBannerDetector(
    config.GREEN_HSV[0], config.GREEN_HSV[1], config.MIN_AREA)

red = RedZoneDetector(
    config.RED_HSV[0][0], config.RED_HSV[0][1],
    config.RED_HSV[1][0], config.RED_HSV[1][1],
    config.MIN_AREA)

nav = CorridorNavigator(config.CORRIDOR_TOLERANCE,
                        config.CORRIDOR_TOLERANCE)

# ── Mission ───────────────────────────────────────────────────────────────────
mission = MissionController(
    target_qr       = "DROP_ZONE_A",
    match_threshold = config.MATCH_THRESHOLD)

# ── Planner — crash-safe build ────────────────────────────────────────────────
planner = None
if config.PLAN_ENABLED and PLANNER_AVAILABLE:
    try:
        planner = build_planner(
            strategy = config.PLAN_STRATEGY,
            rows     = config.PLAN_ROWS,
            hold_sec = config.PLAN_WAYPOINT_HOLD_SEC)
        logger.info(
            f"[MAIN] Planner active  strategy={planner.name}  "
            f"waypoints={len(getattr(planner, 'waypoints', []))}")
    except Exception as e:
        logger.exception(f"[MAIN] Planner construction failed: {e}")
        planner = None
else:
    if not config.PLAN_ENABLED:
        logger.info("[MAIN] PLAN_ENABLED=False — planner disabled")

# ── State ─────────────────────────────────────────────────────────────────────
FCX             = config.RESOLUTION[0] // 2
FCY             = config.RESOLUTION[1] // 2
frame_count     = 0
template_loaded = False
target_data: 'TargetData' = None
match_result              = None
start_qr_data:  str       = ""
debug_template_saved      = False
debug_candidate_saved     = False
plan_early_exit_done      = False

# Temporal smoothing
SMOOTHING_WINDOW = 3
match_history    = []

# Pre-allocate edge overlay buffer
_edge_overlay = np.zeros(
    (config.RESOLUTION[1], config.RESOLUTION[0], 3), dtype=np.uint8)

try:
    while True:
        frame_count += 1
        now = datetime.now().strftime("%H:%M:%S.%f")

        # ── Capture (USB MJPG → BGR, undistorted via cached remap) ───────────
        ok, frame_bgr = cap.read()
        if not ok:
            logger.warning("[MAIN] cap.read() failed — skipping frame")
            continue
        frame_raw = cv2.remap(frame_bgr, _map1, _map2, cv2.INTER_LINEAR)
        frame     = frame_raw.copy()

        # ── Detection ─────────────────────────────────────────────────────────
        qr_results    = qr.detect(frame_raw)
        green_results = green.detect(frame_raw)
        red_results   = red.detect(frame_raw)
        nav_output    = nav.detect(frame_raw)
        center_line, deviation = (nav_output
                                  if isinstance(nav_output, tuple)
                                  else (None, 0))

        # ── STEP 1 — Load template from FIRST QR (once) ───────────────────────
        if qr_results and not template_loaded:
            first_payload = qr_results[0][0]
            first_crop    = cv2.cvtColor(qr_results[0][3], cv2.COLOR_RGB2BGR) \
                if qr_results[0][3] is not None else None

            # Save first QR ALWAYS (even if decoding fails later)
            if config.SAVE_FIRST_QR:
                out.save_first_qr(first_crop, first_payload)

            logger.info(
                f"[DEBUG] First QR payload preview: "
                f"'{first_payload[:80]}{'…' if len(first_payload) > 80 else ''}'  "
                f"len={len(first_payload)}  "
                f"starts_with_data={first_payload.startswith('data:')}")

            extracted = qr.load_target_from_qr_payload(first_payload)
            if extracted is not None and extracted.is_valid():
                target_data     = extracted
                template_loaded = True
                start_qr_data   = first_payload

                if config.SAVE_TEMPLATE_QR:
                    out.save_template(
                        target_data.template_qr_raw,
                        target_data.template_qr_image)

                logger.info(
                    f"[MAIN] ✓ Target template ready  "
                    f"shape={target_data.template_qr_image.shape}  "
                    f"payload_bytes={target_data.payload_size}")
            else:
                logger.warning(
                    "[MAIN] ✗ Could not decode START QR payload — "
                    "will retry on next QR detection.")

        # ── STEP 2 — Per-frame visual matching (for HUD + smoothing) ──────────
        match_result    = None
        skipped_blurry  = 0
        any_match_this_frame = False

        if template_loaded and target_data is not None and qr_results:
            best_found = False
            best_conf  = 0.0
            best_entry = None

            for entry in qr_results:
                data_string, (cxq, cyq), rect, raw_crop, binary_crop = entry

                if data_string == start_qr_data:
                    continue

                # Blur guard on the QR crop (Laplacian variance)
                if raw_crop is not None:
                    _g = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY) \
                        if raw_crop.ndim == 3 else raw_crop
                    if cv2.Laplacian(_g, cv2.CV_64F).var() < config.SHARPNESS_THRESHOLD:
                        skipped_blurry += 1
                        continue

                found, conf = qr.match_against_target(
                    binary_crop, threshold=config.MATCH_THRESHOLD)

                if found:
                    any_match_this_frame = True

                # Save candidate (throttled)
                if config.SAVE_CANDIDATES:
                    out.save_candidate(
                        cv2.cvtColor(raw_crop, cv2.COLOR_RGB2BGR),
                        binary_crop,
                        frame_index = frame_count,
                        confidence  = conf,
                        matched     = found,
                        sharpness   = 0.0)

                logger.info(
                    f"[MAIN] {'✓' if found else '✗'} candidate  "
                    f"conf={conf:.3f}  threshold={config.MATCH_THRESHOLD}")

                if conf > best_conf:
                    best_conf  = conf
                    best_found = found
                    best_entry = entry

            # Temporal smoothing
            if best_entry is not None:
                match_history.append((best_found, best_conf))
                match_history = match_history[-SMOOTHING_WINDOW:]

                confirmed = (len(match_history) == SMOOTHING_WINDOW
                             and all(m[0] for m in match_history))
                avg_conf  = sum(m[1] for m in match_history) / len(match_history)

                _, (cxq, cyq), _, _, _ = best_entry
                match_result = MatchResult(
                    match_found = confirmed,
                    confidence  = round(avg_conf, 4),
                    center      = (cxq, cyq),
                    angle       = 0.0,
                    deviation_x = cxq - FCX,
                    deviation_y = cyq - FCY,
                )

                if confirmed:
                    logger.info(
                        f"[MAIN] ✓✓✓ CONFIRMED MATCH over "
                        f"{SMOOTHING_WINDOW} frames  avg_conf={avg_conf:.3f}")

        # ── Planner step (crash-safe) ─────────────────────────────────────────
        current_wp = None
        if planner is not None:
            try:
                # Early-exit option: stop advancing once we've confirmed match
                if (config.PLAN_EARLY_EXIT_ON_QR
                        and match_result is not None
                        and match_result.match_found
                        and not plan_early_exit_done):
                    logger.info(
                        "[PLANNER] Early exit — confirmed match, freezing path")
                    plan_early_exit_done = True

                if not plan_early_exit_done:
                    current_wp = planner.advance()
                else:
                    current_wp = planner.current_waypoint()
            except Exception as e:
                logger.exception(f"[PLANNER] advance failed: {e}")
                planner = None   # disable for the rest of the run

        # ── Mission logic ─────────────────────────────────────────────────────
        command    = mission.process(
            qr_results, green_results, red_results, deviation)
        debug_info = mission.get_debug_info()

        # ── Edge overlay ──────────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        _edge_overlay[:] = 0
        _edge_overlay[:, :, 0] = edges
        cv2.addWeighted(frame, 1.0, _edge_overlay, 0.35, 0, dst=frame)

        # ── Draw detector overlays ────────────────────────────────────────────
        qr.draw(frame, qr_results)
        green.draw(frame, green_results)
        red.draw(frame, red_results)
        nav.draw(frame, center_line, deviation)

        if center_line is not None:
            cv2.circle(frame, (center_line, FCY), 6, (0, 0, 255), -1)
            cv2.line(frame, (FCX, FCY), (center_line, FCY), (255, 0, 255), 2)

        direction = ("Centered"
                     if abs(deviation) < 20
                     else "MOVE RIGHT" if deviation > 0 else "MOVE LEFT")

        cv2.line(frame, (FCX, 0), (FCX, config.RESOLUTION[1]),
                 config.COLOR_CENTER, 1)

        # ── Draw planner path (crash-safe) ────────────────────────────────────
        if planner is not None and PLANNER_AVAILABLE:
            try:
                draw_path(
                    frame, planner,
                    margin        = config.PLAN_VISUAL_MARGIN_PX,
                    color_pending = config.PLAN_COLOR_PENDING,
                    color_visited = config.PLAN_COLOR_VISITED,
                    color_current = config.PLAN_COLOR_CURRENT,
                    color_path    = config.PLAN_COLOR_PATH)
            except Exception as e:
                logger.debug(f"[PLANNER] draw failed: {e}")

        # ── HUD overlay ───────────────────────────────────────────────────────
        tmpl_text  = "LOADED" if template_loaded else "WAITING FOR START QR"
        tmpl_color = (0, 255, 0) if template_loaded else (0, 165, 255)
        cv2.putText(frame, f"TEMPLATE: {tmpl_text}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tmpl_color, 2)

        cv2.putText(frame, f"STATE: {debug_info['state']}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2)

        cmd_color = ((0, 0, 255)
                     if any(k in command for k in ("ESCAPE", "FORCE", "DROP"))
                     else (0, 255, 255))
        cv2.putText(frame, f"CMD: {command}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2)

        if match_result is not None:
            conf_color = (0, 255, 0) if match_result.match_found else (0, 0, 255)
            match_text = (
                f"MATCH: {'YES' if match_result.match_found else 'NO'}  "
                f"CONF: {match_result.confidence:.2f}")
            cv2.putText(frame, match_text,
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)

            if match_result.match_found:
                cx_m, cy_m = match_result.center
                cv2.drawMarker(frame, (cx_m, cy_m),
                               (0, 255, 0), cv2.MARKER_STAR, 20, 2)
                cv2.putText(frame, "TARGET FOUND",
                            (cx_m - 60, cy_m - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, direction,
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_CENTER, 2)

        cv2.putText(frame, f"QR: {len(qr_results)}  BLURRY: {skipped_blurry}",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if template_loaded and target_data is not None:
            cv2.putText(frame,
                        f"START PAYLOAD: {target_data.payload_size}B",
                        (10, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        if planner is not None and current_wp is not None:
            cv2.putText(frame,
                        f"WP: {current_wp[0]:.2f},{current_wp[1]:.2f}  "
                        f"rem={planner.remaining()}",
                        (10, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        config.PLAN_COLOR_PATH, 1)

        if debug_info.get('in_red_zone', False):
            cv2.rectangle(frame, (5, 250), (ACTUAL_W - 5, 300), (0, 0, 220), -1)
            cv2.putText(frame, "!!! RED ZONE !!!",
                        (ACTUAL_W // 2 - 140, 286),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"FRAME: {frame_count}",
                    (10, config.RESOLUTION[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ── Logging (every LOG_INTERVAL frames) ───────────────────────────────
        if frame_count % LOG_INTERVAL == 0:
            # QR log
            if qr_results:
                for ds, (cxq, cyq), rect, raw_crop, _, is_sharp in qr_results:
                    has_img  = 1 if raw_crop is not None else 0
                    sharp    = 1 if is_sharp else 0
                    ds_short = (ds[:40] + "…") if len(ds) > 40 else ds
                    qr_log.write(
                        f"{now},QR,1,{cxq},{cyq},{ds_short},"
                        f"{has_img},sharp={sharp}\n")
            else:
                qr_log.write(f"{now},QR,0,0,0,None,0,sharp=0\n")

            # Match log
            if match_result is not None and match_result.confidence > 0:
                match_log.write(
                    f"{now},"
                    f"{match_result.match_found},"
                    f"{match_result.confidence:.4f},"
                    f"{match_result.angle:.2f},"
                    f"{match_result.center}\n")

            # Green log
            if green_results:
                for entry in green_results:
                    try:
                        (cxg, cyg), area = entry
                    except Exception:
                        cxg, cyg, area = 0, 0, 0
                    green_log.write(f"{now},GREEN,1,{cxg},{cyg},{area}\n")
            else:
                green_log.write(f"{now},GREEN,0,0,0,0\n")

            # Red log
            if red_results:
                for entry in red_results:
                    try:
                        cxr, cyr, ar = entry
                    except Exception:
                        cxr, cyr, ar = 0, 0, 0
                    red_log.write(f"{now},RED,1,{cxr},{cyr},{ar}\n")
            else:
                red_log.write(f"{now},RED,0,0,0,0\n")

            # Nav log
            nav_log.write(f"{now},NAV,{center_line},{deviation}\n")

            # Mission log
            mission_log.write(
                f"{now},{debug_info['state']},{command},"
                f"timer={debug_info.get('timer','-')},"
                f"red_zone={debug_info.get('in_red_zone', False)}\n")

            # Plan log
            if planner is not None:
                wp = planner.current_waypoint()
                plan_log.write(
                    f"{now},{planner.name},"
                    f"idx={getattr(planner, '_index', '?')},"
                    f"rem={planner.remaining()},"
                    f"wp={wp},"
                    f"early_exit={plan_early_exit_done}\n")
            else:
                plan_log.write(f"{now},disabled\n")

            # Console summary
            conf_val = match_result.confidence if match_result else 0.0
            found    = match_result.match_found if match_result else False
            logger.info(
                f"[FRAME {frame_count}] "
                f"[{debug_info['state']}] {command} | "
                f"QR={len(qr_results)} blurry={skipped_blurry} "
                f"conf={conf_val:.2f} match={found} | "
                f"plan_rem={planner.remaining() if planner else '-'}")

        # ── Output ────────────────────────────────────────────────────────────
        writer.write(frame)
        cv2.imshow("Mission Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logger.info("Manual termination requested (Ctrl+C).")
except Exception as e:
    logger.exception(f"Main loop error: {e}")

finally:
    cap.release()
    writer.release()

    # Persist plan
    if planner is not None and PLANNER_AVAILABLE:
        try:
            save_plan(planner, out.dir_planning)
        except Exception as e:
            logger.debug(f"[PLANNER] save_plan failed: {e}")

    for f in (qr_log, match_log, green_log,
              red_log, nav_log, mission_log, plan_log):
        try:
            f.close()
        except Exception:
            pass

    cv2.destroyAllWindows()
    logger.info(f"✓ Run dir : {out.run_dir}")
    logger.info(f"✓ Video   : {video_path}")
    logger.info(f"✓ Logs    : {out.dir_logs}")
    logger.info(f"✓ Plan    : {out.dir_planning}")
    logger.info("✓ Shutdown complete")