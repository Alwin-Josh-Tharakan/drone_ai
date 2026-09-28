#!/usr/bin/env python3
"""
qr_module.py — single-import facade for the QR stage (drone-hardened).

  * Per-detection STATUS labels (no more misleading "SKIPPED (BLURRY)").
  * START QR search tries EVERY detection, not just detections[0].
  * Full-res pyzbar retry when half-res finds nothing (dense START QRs).
  * Sharpness gate softened (default 25) and its value shown on-screen.
  * Confirmation = M-of-N frames (3 hits in last 5) → vibration tolerant.
  * Professional HUD: L-brackets, colored labels, confidence bar.

Usage (team main file):
    from qr_module import QRPipeline
    pipe = QRPipeline(frame_size=(ACTUAL_W, ACTUAL_H))
    res  = pipe.process_frame(frame)
    pipe.draw(frame, res)
    if res.confirmed: ...
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    confirm_hits:    int   = 0
    status:          Dict[tuple, Tuple[str, tuple]] = field(default_factory=dict)


class QRPipeline:
    def __init__(self,
                 min_size:            int   = 50,
                 template_size:       tuple = (128, 128),
                 grid_threshold:      float = 0.80,
                 match_threshold:     float = 0.75,
                 sharpness_threshold: float = 25.0,     # was 100 — too strict
                 frame_size:          tuple = (1600, 1300),
                 smoothing_window:    int   = 5,        # N (window)
                 confirm_min_hits:    int   = 3,        # M (hits needed)
                 output_dir:          str   = "./output",
                 output_manager       = None,
                 clahe_limit:         float = 2.0,
                 clahe_tile_size:     tuple = (8, 8),
                 detect_scale:        float = 0.5,
                 fullres_retry:       bool  = True):
        self.qr = QRDetector(min_size=min_size, output_dir=output_dir,
                             template_size=template_size, detect_scale=detect_scale)
        self.matcher = TemplateMatcher(match_threshold=match_threshold,
                                       template_size=template_size,
                                       clahe_limit=clahe_limit,
                                       clahe_tile_size=clahe_tile_size,
                                       frame_size=frame_size)
        self.grid_threshold      = grid_threshold
        self.sharpness_threshold = sharpness_threshold
        self.confirm_window      = smoothing_window
        self.confirm_min_hits    = confirm_min_hits
        self.fullres_retry       = fullres_retry
        self.out                 = output_manager

        self.template_loaded = False
        self.start_qr_data   = ""
        self.target_data: Optional[TargetData] = None
        self._history: List[bool] = []
        self._frame_index = 0

    # ── template loading ────────────────────────────────────────────────
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

    # ── per-frame processing ────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        self._frame_index += 1
        res = FrameResult(frame_index=self._frame_index)

        detections = self.qr.detect(frame)
        # Dense START QRs often fail at half-res → one full-res retry
        if not detections and self.fullres_retry and self.qr.detect_scale < 1.0:
            old = self.qr.detect_scale
            self.qr.detect_scale = 1.0
            detections = self.qr.detect(frame)
            self.qr.detect_scale = old
            if detections:
                logger.info(f"[QRMODULE] full-res retry found {len(detections)} QR(s)")
        res.detections = detections

        # ── START QR acquisition: try EVERY detection, not just [0] ─────
        if not self.template_loaded:
            for data, (cx, cy), rect, raw_crop, _b in detections:
                if QRDetector._extract_base64(data) is None:
                    res.status[rect] = ("not a START QR (plain payload)", (160, 160, 160))
                    continue
                if self.load_start_qr(data, raw_crop=raw_crop):
                    res.status[rect] = ("START QR → TEMPLATE LOADED", (255, 200, 0))
                    break
                res.status[rect] = ("START QR payload FAILED", (0, 165, 255))
            res.template_loaded = self.template_loaded
            res.start_qr_data   = self.start_qr_data
            return res

        # ── candidates ──────────────────────────────────────────────────
        fcx, fcy = frame.shape[1] // 2, frame.shape[0] // 2
        best: Optional[CandidateResult] = None

        for data, (cx, cy), rect, raw_crop, binary_crop in detections:
            if data == self.start_qr_data:
                res.status[rect] = ("START QR (template source)", (255, 200, 0))
                continue

            sharp = 0.0
            if raw_crop is not None:
                g = (cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
                     if raw_crop.ndim == 3 else raw_crop)
                sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())

            if sharp < self.sharpness_threshold:
                res.status[rect] = (f"BLURRY {sharp:.0f}<{self.sharpness_threshold:.0f}",
                                    (128, 128, 128))
                continue

            grid_ok, grid_conf = self.qr.match_against_target(
                binary_crop, threshold=self.grid_threshold)
            mres = self.matcher.match(raw_crop, self.target_data.template_qr_image,
                                      candidate_rect=rect)
            matched = bool(grid_ok and mres.match_found)

            cand = CandidateResult(
                data=data, center=(cx, cy), rect=rect, sharpness=round(sharp, 2),
                grid_conf=grid_conf, grid_match=bool(grid_ok),
                matcher_conf=mres.confidence, matcher_match=mres.match_found,
                matched=matched, deviation_x=cx - fcx, deviation_y=cy - fcy)
            res.candidates.append(cand)
            res.status[rect] = (
                (f"MATCH G:{grid_conf:.2f} M:{mres.confidence:.2f}", (0, 255, 0))
                if matched else
                (f"NO MATCH G:{grid_conf:.2f} M:{mres.confidence:.2f}", (0, 0, 255)))

            if self.out is not None:
                self.out.save_candidate(raw_crop, binary_crop,
                                        frame_index=self._frame_index,
                                        confidence=grid_conf, matched=matched,
                                        sharpness=sharp)
            if best is None or cand.grid_conf > best.grid_conf:
                best = cand

        res.best = best

        # ── M-of-N confirmation (vibration tolerant) ────────────────────
        self._history.append(bool(best.matched) if best else False)
        self._history = self._history[-self.confirm_window:]
        res.confirm_hits = sum(self._history)
        res.confirmed = res.confirm_hits >= self.confirm_min_hits
        if best is not None:
            res.confirmed_conf = round((best.grid_conf + best.matcher_conf) / 2.0, 4)
        if res.confirmed:
            logger.info(f"[QRMODULE] ✓✓✓ CONFIRMED ({res.confirm_hits}/"
                        f"{self.confirm_window}) conf={res.confirmed_conf:.3f}")
        return res

    # ── drawing ─────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, res: FrameResult) -> None:
        h, w = frame.shape[:2]
        fcx, fcy = w // 2, h // 2
        cv2.line(frame, (fcx - 15, fcy), (fcx + 15, fcy), (255, 255, 255), 1)
        cv2.line(frame, (fcx, fcy - 15), (fcx, fcy + 15), (255, 255, 255), 1)

        for data, (cx, cy), rect, _raw, _bin in res.detections:
            x, y, rw, rh = rect
            label, color = res.status.get(rect, ("UNPROCESSED", (128, 128, 128)))

            cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)
            blen = max(12, min(rw, rh) // 5)
            for (px, py), (sx, sy) in (((x, y), (1, 1)), ((x + rw, y), (-1, 1)),
                                       ((x, y + rh), (1, -1)), ((x + rw, y + rh), (-1, -1))):
                cv2.line(frame, (px, py), (px + sx * blen, py), color, 3)
                cv2.line(frame, (px, py), (px, py + sy * blen), color, 3)
            cv2.circle(frame, (cx, cy), 5, color, -1)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        if res.best is not None:
            cx, cy = res.best.center
            cv2.line(frame, (fcx, fcy), (cx, cy), (255, 0, 0), 2)
            cv2.putText(frame, f"dx:{res.best.deviation_x} dy:{res.best.deviation_y}",
                        (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if res.confirmed:
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_STAR, 30, 3)
                cv2.putText(frame, "TARGET CONFIRMED", (cx - 90, cy - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self._draw_hud(frame, res)

    def _draw_hud(self, frame, res):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 165), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        t_color = (0, 255, 0) if res.template_loaded else (0, 165, 255)
        t_text  = "LOADED" if res.template_loaded else "WAITING FOR START QR"
        cv2.putText(frame, f"TEMPLATE: {t_text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_color, 2)
        cv2.putText(frame, f"DETECTIONS: {len(res.detections)}  "
                           f"CANDIDATES: {len(res.candidates)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if res.best is not None:
            avg = (res.best.grid_conf + res.best.matcher_conf) / 2.0
            bar_color = ((0, 255, 0) if avg >= 0.75 else
                         (0, 255, 255) if avg >= 0.50 else (0, 0, 255))
            cv2.putText(frame, f"CONFIDENCE: {avg:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 2)
            bx, by, bw, bh = 20, 100, 340, 20
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * min(1.0, avg)), by + bh), bar_color, -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
            tx = bx + int(bw * 0.75)
            cv2.line(frame, (tx, by - 5), (tx, by + bh + 5), (255, 255, 255), 2)
        else:
            cv2.putText(frame, "CONFIDENCE: N/A", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)

        cv2.putText(frame, f"CONFIRM: {res.confirm_hits}/{self.confirm_min_hits} hits",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if res.confirmed else (200, 200, 200), 2)