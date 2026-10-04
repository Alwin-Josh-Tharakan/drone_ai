#!/usr/bin/env python3
"""
detectors/qr_detector.py
Upgraded with "Structural Awareness":
  - Detects QR structures even if unreadable (Orange box).
  - Filters out floor tiles/shadows using a "Squareness" check.
  - A1, B1, B2 upgrades included.
"""

import base64
import binascii
import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode

logger = logging.getLogger(__name__)


class TargetData:
    __slots__ = ("template_qr_image", "template_qr_raw", "timestamp",
                 "template_filename", "payload_size")

    def __init__(self, template_qr_image: np.ndarray, template_qr_raw: np.ndarray,
                 timestamp: str = "", template_filename: str = "", payload_size: int = 0):
        self.template_qr_image = template_qr_image
        self.template_qr_raw = template_qr_raw
        self.timestamp = timestamp
        self.template_filename = template_filename
        self.payload_size = payload_size

    def is_valid(self) -> bool:
        return self.template_qr_image is not None and self.template_qr_image.size > 0


class QRDetector:
    def __init__(self,
                 min_size: int = 50,
                 output_dir: str = "./output",
                 template_size: tuple = (128, 128),
                 grid_n: int = 29,
                 detect_scale: float = 0.5):

        self.min_size = min_size
        self.output_dir = output_dir
        self.template_size = template_size
        self.grid_n = grid_n
        self.detect_scale = detect_scale

        self.template_dir = os.path.join(output_dir, "mission_data", "target_templates")
        os.makedirs(self.template_dir, exist_ok=True)

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # OpenCV shape detector (finds QR structure without decoding)
        self._cv2_qr = cv2.QRCodeDetector()

        self._target_template: Optional[np.ndarray] = None
        self._target_raw: Optional[np.ndarray] = None
        self._start_qr_data: str = ""

    # ─────────────────────────────────────────────────────────────────────
    # Helper: Filter out floor tiles and shadows
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _is_plausible_qr(pts: np.ndarray) -> bool:
        """Check if 4 points form a plausible QR code shape (not a random tile)."""
        if pts is None or len(pts) != 4:
            return False
        
        # Calculate the length of all 4 sides
        sides = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0])
        ]
        
        min_side = min(sides)
        max_side = max(sides)
        
        # A real QR code has roughly equal sides. 
        # Even with extreme perspective, the ratio shouldn't exceed 3.0.
        if min_side < 20:  # Too small
            return False
        if max_side / min_side > 3.0:  # Too skewed (likely a tile or shadow)
            return False
            
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Step 1 — Template loading
    # ──────────────────────────────────────────────────────────────────────
    def load_target_from_qr_payload(self, qr_payload: str) -> Optional[TargetData]:
        if not qr_payload:
            logger.error("[QR] Empty START QR payload")
            return None

        b64_str = self._extract_base64(qr_payload)
        if b64_str is None:
            return None

        try:
            img_bytes = base64.b64decode(b64_str, validate=False)
        except Exception as e:
            logger.error(f"[QR] base64 decode failed: {e}")
            return None

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if raw_img is None:
            return None

        processed = self._preprocess_template(raw_img, denoise=False)
        if processed is None:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        proc_path = os.path.join(self.template_dir, f"target_proc_{ts}.png")
        try:
            cv2.imwrite(os.path.join(self.template_dir, f"target_raw_{ts}.png"), raw_img)
            cv2.imwrite(proc_path, processed)
        except Exception:
            pass

        self._target_template = processed
        self._target_raw = raw_img.copy()
        self._start_qr_data = qr_payload

        return TargetData(
            template_qr_image=processed, template_qr_raw=raw_img.copy(),
            timestamp=ts, template_filename=os.path.basename(proc_path),
            payload_size=len(img_bytes),
        )

    def get_target_template(self) -> Optional[np.ndarray]:
        return self._target_template

    def get_start_qr_data(self) -> str:
        return self._start_qr_data

    # ──────────────────────────────────────────────────────────────────────
    # Step 2 — Per-frame detection (STRUCTURAL AWARENESS)
    # ──────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Tuple]:
        results = []
        h, w = frame.shape[:2]

        s = self.detect_scale
        if 0 < s < 1.0:
            small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray_small = (cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small)
        inv_s = 1.0 / s if 0 < s < 1.0 else 1.0

        # 1. Find QR structures using OpenCV
        retval, points = self._cv2_qr.detect(gray_small)
        
        if not retval or points is None:
            return results

        # 2. Process each detected structure
        for pts in points:
            # Filter out tiles/shadows
            if not self._is_plausible_qr(pts):
                continue

            # Scale points to FULL resolution
            pts_full = (pts * inv_s).astype(np.float32)
            x_f, y_f, w_f, h_f = cv2.boundingRect(pts_full)
            
            if w_f < self.min_size or h_f < self.min_size:
                continue

            cx = int(x_f + w_f // 2)
            cy = int(y_f + h_f // 2)
            original_rect = (int(x_f), int(y_f), int(w_f), int(h_f))

            # 3. Try to decode with pyzbar on the SMALL crop
            x_s, y_s, w_s, h_s = cv2.boundingRect(pts)
            y_start = max(0, y_s)
            x_start = max(0, x_s)
            y_end = min(y_s + h_s, gray_small.shape[0])
            x_end = min(x_s + w_s, gray_small.shape[1])
            
            crop_for_decode = gray_small[y_start:y_end, x_start:x_end]
            if crop_for_decode.size == 0:
                continue

            decoded_objects = pyzbar_decode(crop_for_decode)
            
            # If pyzbar fails, we mark it as a structural QR (Orange box)
            data = "[QR_STRUCTURE]" 
            if decoded_objects:
                data = decoded_objects[0].data.decode("utf-8", errors="replace")

            # 4. B1 UPGRADE: Perspective Warp
            warp_size = 200
            dst_pts = np.array([[0,0], [warp_size,0], [warp_size,warp_size], [0,warp_size]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts_full, dst_pts)
            raw_crop = cv2.warpPerspective(frame, M, (warp_size, warp_size))

            if raw_crop is None or raw_crop.size == 0:
                continue

            binary_crop = self._preprocess_template(raw_crop, denoise=True)
            if binary_crop is None:
                continue

            results.append((data, (cx, cy), original_rect, raw_crop, binary_crop))

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Step 3 — Visual matching (A1: Data-Only Masking)
    # ──────────────────────────────────────────────────────────────────────
    def match_against_target(self, binary_crop: np.ndarray,
                             threshold: float = 0.80) -> Tuple[bool, float]:
        if self._target_template is None:
            return False, 0.0

        tpl = self._target_template
        if binary_crop.shape != tpl.shape:
            binary_crop = cv2.resize(binary_crop, (tpl.shape[1], tpl.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            _, binary_crop = cv2.threshold(binary_crop, 127, 255, cv2.THRESH_BINARY)

        tpl_grid = self._image_to_module_grid(tpl, self.grid_n)

        margin = max(1, int(self.grid_n * 0.25)) 
        mask = np.ones((self.grid_n, self.grid_n), dtype=bool)
        mask[:margin, :margin] = False
        mask[:margin, -margin:] = False
        mask[-margin:, :margin] = False

        best_conf = 0.0
        for k in range(4):
            rotated = (np.ascontiguousarray(np.rot90(binary_crop, k)) if k > 0 else binary_crop)
            cand_grid = self._image_to_module_grid(rotated, self.grid_n)
            agree = np.count_nonzero((cand_grid == tpl_grid) & mask)
            total = np.count_nonzero(mask)
            conf = agree / total if total > 0 else 0.0
            if conf > best_conf:
                best_conf = conf

        return best_conf >= threshold, round(float(best_conf), 4)

    # ──────────────────────────────────────────────────────────────────────
    # Drawing (HUD) - Orange for structure, Green for match
    # ──────────────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, detections: list) -> None:
        for entry in detections:
            data, (cx, cy), rect, raw_crop, binary_crop = entry
            rx, ry, rw, rh = rect

            # Default to Orange (Structure detected)
            box_color = (0, 165, 255) 
            status = "QR STRUCTURE"

            if binary_crop is not None and self._target_template is not None:
                found, conf = self.match_against_target(binary_crop)
                if found:
                    box_color = (0, 255, 0) # Green if matched
                    status = f"MATCH {conf:.2f}"
                elif data != "[QR_STRUCTURE]":
                    box_color = (0, 0, 255) # Red if decoded but wrong
                    status = f"WRONG QR {conf:.2f}"

            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)
            cv2.circle(frame, (cx, cy), 5, box_color, -1)

            cv2.putText(frame, status, (rx, max(ry - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers (B2: Module-Centre Sampling)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _image_to_module_grid(binary: np.ndarray, grid_n: int) -> np.ndarray:
        k = 3
        big_size = grid_n * k
        resized = cv2.resize(binary, (big_size, big_size), interpolation=cv2.INTER_AREA)
        _, resized = cv2.threshold(resized, 127, 1, cv2.THRESH_BINARY)
        centers = resized[k//2::k, k//2::k]
        return centers.astype(np.uint8)

    @staticmethod
    def _extract_base64(payload: str) -> Optional[str]:
        s = payload.strip()
        if s.startswith("data:") and ";base64," in s:
            s = s.split(";base64,", 1)[1]
        for prefix in ("IMG:", "TARGET:", "TPL:", "B64:"):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        s = s.strip().replace("\n", "").replace("\r", "").replace(" ", "")
        if len(s) < 100:
            return None
        try:
            decoded = base64.b64decode(s, validate=False)
            if (decoded[:4] == b"\x89PNG" or decoded[:3] == b"\xff\xd8\xff"
                    or decoded[:2] == b"BM" or decoded[:4] == b"GIF8"):
                return s
            return None
        except Exception:
            return None

    def _preprocess_template(self, img: np.ndarray, denoise: bool = False) -> Optional[np.ndarray]:
        try:
            gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy())
            if denoise:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                eq = self._clahe.apply(gray)
                binary = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
            else:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            trimmed = self._trim_quiet_zone(binary)
            if trimmed is None or trimmed.size == 0:
                trimmed = binary

            resized = cv2.resize(trimmed, self.template_size, interpolation=cv2.INTER_AREA)
            _, clean = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            return clean
        except Exception as e:
            logger.error(f"[QR] _preprocess_template: {e}")
            return None

    @staticmethod
    def _trim_quiet_zone(binary: np.ndarray) -> Optional[np.ndarray]:
        try:
            inv = cv2.bitwise_not(binary)
            coords = cv2.findNonZero(inv)
            if coords is None:
                return None
            x, y, w, h = cv2.boundingRect(coords)
            m = 2
            x1, y1 = max(0, x - m), max(0, y - m)
            x2 = min(binary.shape[1], x + w + m)
            y2 = min(binary.shape[0], y + h + m)
            return binary[y1:y2, x1:x2]
        except Exception:
            return None