"""
detectors/qr_detector.py
QR detection + pure-visual template matching for the SINOSEE global-shutter
USB camera mission (optimized fork).

Debugged / hardened in this revision
------------------------------------
* detect() is now MULTI-PASS for printed / low-contrast sheets:
    pass 1: downscaled gray           (fast path)
    pass 2: full-res gray             (dense QRs that vanish at half-res)
    pass 3: full-res CLAHE + unsharp  (every enhanced_every frames)
    pass 4: full-res adaptive binary  (every enhanced_every frames)
* New shared binarize_qr(): CLAHE -> global Otsu keeps modules SOLID at any
  module size. The old fixed adaptiveThreshold(block=11) hollowed out modules
  whenever a module was wider than ~11 px (close-up crops), which destroyed
  grid + matcher scores (G~0.57 / M~0.27 on the TRUE target). Adaptive
  threshold is now only a FALLBACK, with a module-scaled block size.
* Module-grid sampling stays vectorised (integral image), bit-exact.
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
    """Immutable mission reference built ONCE from the START QR payload."""

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
    """Detects QR codes via pyzbar and matches them visually against a stored template."""

    def __init__(self,
                 min_size: int = 50,
                 output_dir: str = "./output",
                 template_size: tuple = (128, 128),
                 grid_n: int = 29,
                 detect_scale: float = 0.5):
        self.min_size = min_size
        self.output_dir = output_dir
        self.template_size = template_size
        self.grid_n = grid_n              # QR module grid for similarity
        self.detect_scale = detect_scale  # pyzbar runs on scaled-down gray frame

        self.template_dir = os.path.join(output_dir, "mission_data", "target_templates")
        os.makedirs(self.template_dir, exist_ok=True)

        # CLAHE — shared parameters for noisy-image preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Stored target template (set once by load_target_from_qr_payload)
        self._target_template: Optional[np.ndarray] = None
        self._target_raw: Optional[np.ndarray] = None
        self._start_qr_data: str = ""

        # multi-pass decode support (dense / low-contrast printed QRs)
        self.enhanced_every = 5     # contrast-boosted passes every 5th frame
        self._tick = 0

    # ──────────────────────────────────────────────────────────────────────
    # Step 1 — Template loading (runs ONCE at startup)
    # ──────────────────────────────────────────────────────────────────────

    def load_target_from_qr_payload(self, qr_payload: str) -> Optional[TargetData]:
        """
        Decode *qr_payload* (text from the START QR) into an image and build
        the canonical binary template. Supports pure base64, data URIs and
        custom prefixes ("IMG:", "TARGET:", "TPL:", "B64:").
        """
        if not qr_payload:
            logger.error("[QR] Empty START QR payload")
            return None

        b64_str = self._extract_base64(qr_payload)
        if b64_str is None:
            logger.error(f"[QR] START QR payload not recognised as base64 image "
                         f"(first 80 chars: '{qr_payload[:80]}')")
            return None

        try:
            img_bytes = base64.b64decode(b64_str, validate=False)
        except (binascii.Error, ValueError) as e:
            logger.error(f"[QR] base64 decode failed: {e}")
            return None

        if not img_bytes:
            logger.error("[QR] base64 decoded to empty bytes")
            return None

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if raw_img is None:
            logger.error(f"[QR] cv2.imdecode failed — payload is not a valid image "
                         f"({len(img_bytes)} bytes)")
            return None

        logger.info(f"[QR] Decoded START QR payload → image "
                    f"shape={raw_img.shape} bytes={len(img_bytes)}")

        processed = self._preprocess_template(raw_img, denoise=False)
        if processed is None:
            logger.error("[QR] Target template preprocessing failed")
            return None

        # Persist debug copies
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        proc_path = os.path.join(self.template_dir, f"target_proc_{ts}.png")
        try:
            cv2.imwrite(os.path.join(self.template_dir, f"target_raw_{ts}.png"), raw_img)
            cv2.imwrite(proc_path, processed)
            logger.info(f"[QR] Saved processed target → {proc_path}")
        except Exception as e:
            logger.warning(f"[QR] could not save target copies: {e}")

        self._target_template = processed
        self._target_raw = raw_img.copy()
        self._start_qr_data = qr_payload

        logger.info(f"[QR] ✓ Target template ready shape={processed.shape} "
                    f"unique={np.unique(processed).tolist()}")

        return TargetData(
            template_qr_image=processed,
            template_qr_raw=raw_img.copy(),
            timestamp=ts,
            template_filename=os.path.basename(proc_path),
            payload_size=len(img_bytes),
        )

    def get_target_template(self) -> Optional[np.ndarray]:
        return self._target_template

    def get_start_qr_data(self) -> str:
        return self._start_qr_data

    # ──────────────────────────────────────────────────────────────────────
    # Shared scale-aware binarisation (THE fix for hollow modules)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _odd(n: int) -> int:
        n = int(n)
        return n + 1 if n % 2 == 0 else n

    @staticmethod
    def binarize_qr(gray: np.ndarray, clahe) -> np.ndarray:
        """
        CLAHE → global Otsu (ideal for printed QR under even light).
        Otsu keeps modules SOLID at any module size. Only if the black
        fraction is implausible (heavy shadow / gradient) fall back to
        adaptive threshold with a MODULE-SCALED block (≈1.5 × module px) —
        the old fixed block=11 hollowed out modules on close-up crops.
        """
        eq = clahe.apply(gray)
        _, binary = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frac = float(np.count_nonzero(binary == 0)) / binary.size
        if 0.15 < frac < 0.75:
            return binary
        module_px = max(3.0, min(gray.shape[0], gray.shape[1]) / 37.0)
        block = min(99, max(11, QRDetector._odd(round(module_px * 1.5))))
        return cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, 2)

    # ──────────────────────────────────────────────────────────────────────
    # Step 2 — Per-frame detection (MULTI-PASS decode)
    # ──────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """
        Find QR-shaped regions in *frame* (undistorted USB capture).
        Multi-pass decode for robustness on printed / low-contrast sheets:
          pass 1: downscaled gray           (fast path)
          pass 2: full-res gray             (dense QRs that vanish at half-res)
          pass 3: full-res CLAHE + unsharp  (every enhanced_every frames)
          pass 4: full-res adaptive binary  (every enhanced_every frames)
        Returns list of (data_string, (cx, cy), rect, raw_crop, binary_crop)
        with all coordinates in FULL-frame pixel space.
        """
        results = []
        h, w = frame.shape[:2]
        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame)
        self._tick += 1

        s = self.detect_scale if 0 < self.detect_scale < 1.0 else 1.0
        passes = []
        if s < 1.0:
            passes.append((cv2.resize(gray, (int(w * s), int(h * s)),
                                      interpolation=cv2.INTER_AREA), 1.0 / s))
        passes.append((gray, 1.0))
        if self._tick % max(1, self.enhanced_every) == 0:
            eq = self._clahe.apply(gray)
            unsharp = cv2.addWeighted(
                eq, 1.6, cv2.GaussianBlur(eq, (0, 0), 2.0), -0.6, 0)
            passes.append((unsharp, 1.0))
            passes.append((cv2.adaptiveThreshold(
                eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 10), 1.0))

        objs, inv_s = [], 1.0
        for img, inv in passes:
            objs = pyzbar_decode(img)
            if objs:
                inv_s = inv
                break

        for obj in objs:
            data = obj.data.decode("utf-8", errors="replace") if obj.data else ""
            r = obj.rect
            rw = int(r.width * inv_s)
            rh = int(r.height * inv_s)
            if rw < self.min_size or rh < self.min_size:
                continue
            rx = int(r.left * inv_s)
            ry = int(r.top * inv_s)
            cx = rx + rw // 2
            cy = ry + rh // 2
            rect = (rx, ry, rw, rh)
            raw_crop = self._crop_qr(frame, rect)
            if raw_crop is None:
                continue
            binary_crop = self._preprocess_template(raw_crop, denoise=True)
            if binary_crop is None:
                continue
            results.append((data, (cx, cy), rect, raw_crop, binary_crop))
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Step 3 — Pure visual matching (no payload decoding involved)
    # ──────────────────────────────────────────────────────────────────────

    def match_against_target(self, binary_crop: np.ndarray,
                             threshold: float = 0.80) -> Tuple[bool, float]:
        """
        Visually compare candidate to stored template across 4 rotations.
        Returns (match_found, confidence ∈ [0,1]).
        """
        if self._target_template is None:
            return False, 0.0

        tpl = self._target_template

        if binary_crop.shape != tpl.shape:
            binary_crop = cv2.resize(binary_crop, (tpl.shape[1], tpl.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            _, binary_crop = cv2.threshold(binary_crop, 127, 255, cv2.THRESH_BINARY)

        tpl_grid = self._image_to_module_grid(tpl, self.grid_n)

        best_conf = 0.0
        for k in range(4):
            rotated = (np.ascontiguousarray(np.rot90(binary_crop, k))
                       if k > 0 else binary_crop)
            cand_grid = self._image_to_module_grid(rotated, self.grid_n)
            conf = np.count_nonzero(cand_grid == tpl_grid) / (self.grid_n * self.grid_n)
            if conf > best_conf:
                best_conf = conf

        logger.debug(f"[QR] match best_conf={best_conf:.3f} threshold={threshold:.2f}")
        return best_conf >= threshold, round(float(best_conf), 4)

    # ──────────────────────────────────────────────────────────────────────
    # Drawing (HUD)
    # ──────────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, detections: list) -> None:
        """Full visual aids: bounding box + corner brackets + center dot,
        payload label, match status/confidence and deviation cues."""
        fcx, fcy = frame.shape[1] // 2, frame.shape[0] // 2
        for entry in detections:
            data, (cx, cy), rect, raw_crop, binary_crop = entry
            rx, ry, rw, rh = rect

            if binary_crop is not None and self._target_template is not None:
                found, conf = self.match_against_target(binary_crop)
                box_color = (0, 255, 0) if found else (255, 0, 255)
                status = f"CONF:{conf:.2f}"
            else:
                found = False
                box_color = (255, 0, 255)
                status = "[CROP OK]" if raw_crop is not None else "[NO CROP]"

            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)
            blen = max(8, min(rw, rh) // 5)
            for (px, py), (sx, sy) in (((rx, ry), (1, 1)), ((rx + rw, ry), (-1, 1)),
                                       ((rx, ry + rh), (1, -1)),
                                       ((rx + rw, ry + rh), (-1, -1))):
                cv2.line(frame, (px, py), (px + sx * blen, py), box_color, 3)
                cv2.line(frame, (px, py), (px, py + sy * blen), box_color, 3)

            cv2.circle(frame, (cx, cy), 5, box_color, -1)
            cv2.line(frame, (cx - 12, cy), (cx + 12, cy), box_color, 1)
            cv2.line(frame, (cx, cy - 12), (cx, cy + 12), box_color, 1)
            cv2.line(frame, (fcx, fcy), (cx, cy), (255, 0, 0), 1)

            label = (data[:20] + "…") if len(data) > 20 else (data or "QR")
            cv2.putText(frame, f"QR: {label}", (rx, max(ry - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            cv2.putText(frame, status, (rx, ry + rh + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            if found:
                dev_x, dev_y = cx - fcx, cy - fcy
                cv2.putText(frame, f"TARGET FOUND  dev=({dev_x:+d},{dev_y:+d})",
                            (rx, max(ry - 32, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _image_to_module_grid(binary: np.ndarray, grid_n: int) -> np.ndarray:
        """
        Sample *binary* onto an NxN module grid (vectorised block-mean).
        Bit-exact with the original Python double-loop implementation
        (same floor-based cell boundaries and majority rule), but ~100x faster.
        Returns uint8 grid: 0 = black module, 1 = white module.
        """
        h, w = binary.shape[:2]
        y_edges = (np.arange(grid_n + 1) * h // grid_n)
        x_edges = (np.arange(grid_n + 1) * w // grid_n)

        ii = cv2.integral(binary.astype(np.float64))
        ys = ii[np.ix_(y_edges, x_edges)]
        cell_sum = ys[:-1, :-1] - ys[1:, :-1] - ys[:-1, 1:] + ys[1:, 1:]
        cell_area = np.outer(np.diff(y_edges), np.diff(x_edges)).astype(np.float64)
        cell_mean = cell_sum / np.maximum(cell_area, 1)
        return (cell_mean >= 127).astype(np.uint8)

    @staticmethod
    def _extract_base64(payload: str) -> Optional[str]:
        """Strip wrapper / data URI and return bare base64 (None if not an image)."""
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

    @staticmethod
    def _crop_qr(frame: np.ndarray, rect: tuple) -> Optional[np.ndarray]:
        try:
            x, y, w, h = rect
            margin = max(4, int(min(w, h) * 0.08))
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            crop = frame[y1:y2, x1:x2]
            return crop.copy() if crop.size > 0 else None
        except Exception as e:
            logger.debug(f"[QR] _crop_qr: {e}")
            return None

    def _preprocess_template(self, img: np.ndarray,
                             denoise: bool = False) -> Optional[np.ndarray]:
        """
        Canonical QR → binary pipeline.
        CLEAN path (denoise=False): gray → Otsu → trim → resize → re-binarise
        NOISY path (denoise=True):  gray → blur → binarize_qr → trim → resize
        """
        try:
            gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if img.ndim == 3 else img.copy())
            if denoise:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                binary = self.binarize_qr(gray, self._clahe)
            else:
                _, binary = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            trimmed = self._trim_quiet_zone(binary)
            if trimmed is None or trimmed.size == 0:
                trimmed = binary

            resized = cv2.resize(trimmed, self.template_size,
                                 interpolation=cv2.INTER_AREA)
            _, clean = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            return clean
        except Exception as e:
            logger.error(f"[QR] _preprocess_template: {e}")
            return None

    @staticmethod
    def _trim_quiet_zone(binary: np.ndarray) -> Optional[np.ndarray]:
        """Remove white border so candidate and template both contain only modules."""
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