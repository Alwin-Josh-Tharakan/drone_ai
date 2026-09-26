#!/usr/bin/env python3
"""
qr_module.py — SINGLE-IMPORT facade for the QR stage of the mission.
(Upgraded with professional visual cues, L-brackets, and confidence meter)
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detectors.qr_detector      import QRDetector, TargetData
from detectors.template_matcher import TemplateMatcher, MatchResult

logger = logging.getLogger(__name__)

@dataclass
class CandidateResult:
    data:         str
    center:       Tuple[int, int]
    rect:         Tuple[int, int, int, int]
    sharpness:    float
    grid_conf:    float
    grid_match:   bool
    matcher_conf: float
    matcher_match: bool
    matched:      bool
    deviation_x:  int = 0
    deviation_y:  int = 0

@dataclass
class FrameResult:
    frame_index:     int   = 0
    template_loaded: bool  = False
    start_qr_data:   str   = ""
    detections:      list  = field(default_factory=list)
    candidates:      List[CandidateResult] = field(default_factory=list)
    best:            Optional[CandidateResult] = None
    confirmed:       bool  = False
    confirmed_conf:  float = 0.0

class QRPipeline:
    def __init__(self,
                 min_size:            int   = 50,
                 template_size:       tuple = (128, 128),
                 grid_threshold:      float = 0.80,
                 match_threshold:     float = 0.75,
                 sharpness_threshold: float = 100.0,
                 frame_size:          tuple = (1600, 1300),
                 smoothing_window:    int   = 3,
                 output_dir:          str   = "./output",
                 output_manager       = None,
                 clahe_limit:         float = 2.0,
                 clahe_tile_size:     tuple = (8, 8),
                 detect_scale:        float = 0.5):
        
        self.qr = QRDetector(min_size=min_size, output_dir=output_dir,
                             template_size=template_size, detect_scale=detect_scale)
        self.matcher = TemplateMatcher(match_threshold=match_threshold,
                                       template_size=template_size,
                                       clahe_limit=clahe_limit,
                                       clahe_tile_size=clahe_tile_size,
                                       frame_size=frame_size)
        self.grid_threshold      = grid_threshold
        self.sharpness_threshold = sharpness_threshold
        self.smoothing_window    = smoothing_window
        self.out                 = output_manager

        self.template_loaded = False
        self.start_qr_data   = ""
        self.target_data: Optional[TargetData] = None
        self._history: List[bool] = []
        self._frame_index = 0

    def load_start_qr(self, payload: str, raw_crop=None) -> bool:
        if self.out is not None and raw_crop is not None:
            self.out.save_first_qr(raw_crop, payload)
        td = self.qr.load_target_from_qr_payload(payload)
        if td is not None and td.is_valid():
            self.target_data     = td
            self.template_loaded = True
            self.start_qr_data   = payload
            if self.out is not None:
                self.out.save_template(td.template_qr_raw, td.template_qr_image)
            logger.info("[QRMODULE] ✓ Template loaded from START QR")
            return True
        logger.warning("[QRMODULE] ✗ START QR payload not usable")
        return False

    def reset(self) -> None:
        self.template_loaded = False
        self.start_qr_data   = ""
        self.target_data     = None
        self._history.clear()

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        self._frame_index += 1
        res = FrameResult(frame_index=self._frame_index)

        detections = self.qr.detect(frame)
        res.detections = detections

        # STEP 1 — first QR ever seen = START QR
        if detections and not self.template_loaded:
            first_payload = detections[0][0]
            first_crop    = detections[0][3]
            self.load_start_qr(first_payload, raw_crop=first_crop)
            res.template_loaded = self.template_loaded
            res.start_qr_data   = self.start_qr_data
            return res

        res.template_loaded = self.template_loaded
        res.start_qr_data   = self.start_qr_data
        if not self.template_loaded:
            return res

        # STEP 2 — candidates
        fcx, fcy = frame.shape[1] // 2, frame.shape[0] // 2
        best: Optional[CandidateResult] = None

        for data, (cx, cy), rect, raw_crop, binary_crop in detections:
            if data == self.start_qr_data:
                continue

            sharp = 0.0
            if raw_crop is not None:
                g = (cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY) if raw_crop.ndim == 3 else raw_crop)
                sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
                if sharp < self.sharpness_threshold:
                    continue

            grid_ok, grid_conf = self.qr.match_against_target(binary_crop, threshold=self.grid_threshold)
            mres = self.matcher.match(raw_crop, self.target_data.template_qr_image, candidate_rect=rect)

            matched = bool(grid_ok and mres.match_found)
            cand = CandidateResult(
                data=data, center=(cx, cy), rect=rect, sharpness=round(sharp, 2),
                grid_conf=grid_conf, grid_match=bool(grid_ok),
                matcher_conf=mres.confidence, matcher_match=mres.match_found,
                matched=matched, deviation_x=cx - fcx, deviation_y=cy - fcy)
            res.candidates.append(cand)

            # SAVE EVERY CANDIDATE (Throttle is controlled by config.py)
            if self.out is not None:
                self.out.save_candidate(raw_crop, binary_crop, frame_index=self._frame_index,
                                        confidence=grid_conf, matched=matched, sharpness=sharp)
            if best is None or cand.grid_conf > best.grid_conf:
                best = cand

        res.best = best

        # STEP 3 — temporal smoothing
        self._history.append(bool(best.matched) if best else False)
        self._history = self._history[-self.smoothing_window:]
        res.confirmed = (len(self._history) == self.smoothing_window and all(self._history))
        if best is not None:
            res.confirmed_conf = round((best.grid_conf + best.matcher_conf) / 2.0, 4)
        if res.confirmed:
            logger.info(f"[QRMODULE] ✓✓✓ CONFIRMED match  conf={res.confirmed_conf:.3f}")
        return res

    # ──────────────────────────────────────────────────────────────────────
    # UPGRADED VISUAL CUES & HUD
    # ──────────────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, res: FrameResult) -> None:
        h, w = frame.shape[:2]
        fcx, fcy = w // 2, h // 2
        
        # 1. Frame Center Crosshair
        cv2.line(frame, (fcx - 15, fcy), (fcx + 15, fcy), (255, 255, 255), 1)
        cv2.line(frame, (fcx, fcy - 15), (fcx, fcy + 15), (255, 255, 255), 1)

        # 2. Draw Detections (Boxes, Brackets, Labels)
        for data, (cx, cy), rect, _raw, _bin in res.detections:
            x, y, rw, rh = rect
            is_start = (res.start_qr_data and data == res.start_qr_data)
            
            if is_start:
                color = (255, 200, 0)  # Cyan
                label = "START QR (TEMPLATE SOURCE)"
            else:
                cand = next((c for c in res.candidates if c.rect == rect), None)
                if cand and cand.matched:
                    color = (0, 255, 0)  # Green
                    label = f"MATCH [G:{cand.grid_conf:.2f} M:{cand.matcher_conf:.2f}]"
                elif cand:
                    color = (0, 0, 255)  # Red
                    label = f"NO MATCH [G:{cand.grid_conf:.2f} M:{cand.matcher_conf:.2f}]"
                else:
                    color = (128, 128, 128) # Gray
                    label = "SKIPPED (BLURRY)"

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)
            
            # L-brackets (Clean visual cue)
            blen = max(12, min(rw, rh) // 5)
            for (px, py), (sx, sy) in (((x, y), (1, 1)), ((x + rw, y), (-1, 1)),
                                        ((x, y + rh), (1, -1)), ((x + rw, y + rh), (-1, -1))):
                cv2.line(frame, (px, py), (px + sx * blen, py), color, 3)
                cv2.line(frame, (px, py), (px, py + sy * blen), color, 3)
                
            # Center dot
            cv2.circle(frame, (cx, cy), 5, color, -1)
            
            # Label with solid background for readability
            font_scale = 0.55
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # 3. Deviation Line for Best Candidate
        if res.best is not None:
            cx, cy = res.best.center
            cv2.line(frame, (fcx, fcy), (cx, cy), (255, 0, 0), 2)
            cv2.putText(frame, f"dx:{res.best.deviation_x} dy:{res.best.deviation_y}", 
                        (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
            if res.confirmed:
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_STAR, 30, 3)
                cv2.putText(frame, "TARGET CONFIRMED", (cx - 90, cy - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 4. HUD Overlay
        self._draw_hud(frame, res, w, h)

    def _draw_hud(self, frame, res, w, h):
        # Semi-transparent black background for HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Template Status
        t_color = (0, 255, 0) if res.template_loaded else (0, 165, 255)
        t_text = "LOADED" if res.template_loaded else "WAITING FOR START QR"
        cv2.putText(frame, f"TEMPLATE: {t_text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_color, 2)

        # Stats
        cv2.putText(frame, f"DETECTIONS: {len(res.detections)}  CANDIDATES: {len(res.candidates)}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Confidence Meter (Visual Cue)
        if res.best is not None:
            avg_conf = (res.best.grid_conf + res.best.matcher_conf) / 2.0
            
            # Color based on confidence
            if avg_conf >= 0.75:
                bar_color = (0, 255, 0)  # Green
            elif avg_conf >= 0.50:
                bar_color = (0, 255, 255) # Yellow
            else:
                bar_color = (0, 0, 255)  # Red
                
            cv2.putText(frame, f"CONFIDENCE: {avg_conf:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 2)
            
            # Draw Bar
            bar_x, bar_y, bar_w, bar_h = 20, 100, 340, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * min(1.0, avg_conf))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
            
            # Threshold line (Visual indicator of the passing mark)
            thresh_x = bar_x + int(bar_w * 0.75) 
            cv2.line(frame, (thresh_x, bar_y - 5), (thresh_x, bar_y + bar_h + 5), (255, 255, 255), 2)
        else:
            cv2.putText(frame, "CONFIDENCE: N/A", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)

        # Confirmation Status
        if res.confirmed:
            cv2.putText(frame, ">>> TARGET CONFIRMED <<<", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)