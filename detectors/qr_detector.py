# detectors/qr_detector.py

from pyzbar.pyzbar import decode as pyzbar_decode
import cv2
import numpy as np
import os
import base64
import binascii
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TargetData:
    """
    Immutable mission reference built ONCE from the START QR payload.
    template_qr_image : binary image at TEMPLATE_SIZE — used for matching
    template_qr_raw   : decoded raw image kept for debug saves
    """
    template_qr_image: np.ndarray
    template_qr_raw:   np.ndarray
    timestamp:         str = ""
    template_filename: str = ""
    payload_size:      int = 0

    def is_valid(self) -> bool:
        return (self.template_qr_image is not None
                and self.template_qr_image.size > 0)


class QRDetector:
    """
    Detects QR codes via pyzbar and matches them visually against a stored
    template.

    WORKFLOW
    --------
    1. STARTUP (once):
       The START QR payload is base64-encoded image bytes of the target
       QR.  load_target_from_qr_payload() decodes it, binarises it and
       stores it as the canonical template.

    2. PER FRAME:
       detect() runs pyzbar purely to LOCATE QR-shaped regions in the
       frame.  The decoded text of these candidates is discarded — we
       only keep their bounding box.  Each candidate is cropped, binarised,
       and compared to the stored template via module-grid similarity.

    MATCHING — pure visual, no payload comparison
    --------
    For each rotation k ∈ {0°, 90°, 180°, 270°}:
        sample candidate and template onto 29×29 module grid
        count agreeing bits
    confidence = best agreement ratio across all rotations
    """

    def __init__(self,
                 min_size:      int   = 50,
                 output_dir:    str   = "/home/pigec/drone_mission/output",
                 template_size: tuple = (128, 128),
                 grid_n:        int   = 29):

        self.min_size      = min_size
        self.output_dir    = output_dir
        self.template_size = template_size
        self.grid_n        = grid_n           # QR module grid for similarity
        self.qr_save_dir   = None

        self.template_dir = os.path.join(
            output_dir, "mission_data", "target_templates")
        os.makedirs(self.template_dir, exist_ok=True)

        # CLAHE — shared parameters for noisy-image preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Stored target template (set once by load_target_from_qr_payload)
        self._target_template: Optional[np.ndarray] = None
        self._target_raw:      Optional[np.ndarray] = None
        self._start_qr_data:   str                  = ""

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — Template loading (runs ONCE at startup)
    # ──────────────────────────────────────────────────────────────────────────

    def load_target_from_qr_payload(self,
                                     qr_payload: str
                                     ) -> Optional[TargetData]:
        """
        Decode *qr_payload* (text decoded from the START QR) into an image
        and build the canonical binary template from it.

        Supports:
            * pure base64
            * data URI ("data:image/png;base64,...")
            * custom prefixes ("IMG:", "TARGET:", "TPL:", "B64:")
        """
        if not qr_payload:
            logger.error("[QR] Empty START QR payload")
            return None

        # 1. Strip wrapper / data URI
        b64_str = self._extract_base64(qr_payload)
        if b64_str is None:
            logger.error(
                f"[QR] START QR payload not recognised as base64 image  "
                f"(first 80 chars: '{qr_payload[:80]}')")
            return None

        # 2. Base64 → bytes
        try:
            img_bytes = base64.b64decode(b64_str, validate=False)
        except (binascii.Error, ValueError) as e:
            logger.error(f"[QR] base64 decode failed: {e}")
            return None

        if not img_bytes:
            logger.error("[QR] base64 decoded to empty bytes")
            return None

        # 3. Bytes → image
        arr     = np.frombuffer(img_bytes, dtype=np.uint8)
        raw_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if raw_img is None:
            logger.error(
                f"[QR] cv2.imdecode failed — payload is not a valid image  "
                f"({len(img_bytes)} bytes)")
            return None

        logger.info(
            f"[QR] Decoded START QR payload → image  "
            f"shape={raw_img.shape}  bytes={len(img_bytes)}")

        # 4. Preprocess clean image → binary template (no denoise — it's clean)
        processed = self._preprocess_template(raw_img, denoise=False)
        if processed is None:
            logger.error("[QR] Target template preprocessing failed")
            return None

        # 5. Persist debug copies
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path  = os.path.join(self.template_dir, f"target_raw_{ts}.png")
        proc_path = os.path.join(self.template_dir, f"target_proc_{ts}.png")
        try:
            cv2.imwrite(raw_path, raw_img)
            cv2.imwrite(proc_path, processed)
            logger.info(f"[QR] Saved raw target       → {raw_path}")
            logger.info(f"[QR] Saved processed target → {proc_path}")
        except Exception as e:
            logger.warning(f"[QR] could not save target copies: {e}")

        self._target_template = processed
        self._target_raw      = raw_img.copy()
        self._start_qr_data   = qr_payload

        self._write_template_metadata(
            ts, os.path.basename(proc_path),
            f"<embedded_image:{len(img_bytes)}bytes>")

        logger.info(
            f"[QR] ✓ Target template ready  "
            f"shape={processed.shape}  "
            f"unique={np.unique(processed).tolist()}")

        return TargetData(
            template_qr_image = processed,
            template_qr_raw   = raw_img.copy(),
            timestamp         = ts,
            template_filename = os.path.basename(proc_path),
            payload_size      = len(img_bytes),
        )

    def get_target_template(self) -> Optional[np.ndarray]:
        return self._target_template

    def get_start_qr_data(self) -> str:
        return self._start_qr_data

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — Per-frame detection (pyzbar used ONLY to locate, not decode)
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """
        Find QR-shaped regions in *frame*.

        pyzbar's decoded text is preserved in the tuple for logging/HUD only —
        it is NOT used by the matcher.

        Returns
        -------
        list of (data_string, (cx, cy), rect, raw_crop, binary_crop)
        """
        results = []
        decoded = pyzbar_decode(frame)

        for obj in decoded:
            # Keep decoded text only for display / logging — matcher ignores it
            data = obj.data.decode('utf-8', errors='replace') if obj.data else ""
            r    = obj.rect

            if r.width < self.min_size or r.height < self.min_size:
                logger.debug(
                    f"[QR] Skipped small detection: {r.width}×{r.height}")
                continue

            cx   = r.left + r.width  // 2
            cy   = r.top  + r.height // 2
            rect = (r.left, r.top, r.width, r.height)

            raw_crop = self._crop_qr(frame, r)
            if raw_crop is None:
                logger.warning("[QR] Failed to crop QR region — skipping")
                continue

            binary_crop = self._preprocess_template(raw_crop, denoise=True)
            if binary_crop is None:
                logger.warning("[QR] Binary preprocessing failed — skipping")
                continue

            results.append((data, (cx, cy), rect, raw_crop, binary_crop))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — Pure visual matching (no decoding involved)
    # ──────────────────────────────────────────────────────────────────────────

    def match_against_target(self,
                             binary_crop: np.ndarray,
                             threshold:   float = 0.80
                             ) -> Tuple[bool, float]:
        """
        Visually compare candidate to stored template.

        Method
        ------
        For each rotation k ∈ {0°, 90°, 180°, 270°}:
            sample candidate and template onto grid_n × grid_n module grid
            confidence_k = fraction of agreeing module bits
        Return the best confidence across all 4 rotations.

        Returns (match_found, confidence ∈ [0,1]).
        """
        if self._target_template is None:
            return False, 0.0

        tpl = self._target_template

        # Ensure same size
        if binary_crop.shape != tpl.shape:
            binary_crop = cv2.resize(
                binary_crop, (tpl.shape[1], tpl.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            _, binary_crop = cv2.threshold(
                binary_crop, 127, 255, cv2.THRESH_BINARY)

        # Pre-compute template module grid once per call
        tpl_grid = self._image_to_module_grid(tpl, self.grid_n)

        best_conf = 0.0
        best_rot  = 0
        for k in range(4):
            rotated = (np.ascontiguousarray(np.rot90(binary_crop, k))
                       if k > 0 else binary_crop)
            cand_grid = self._image_to_module_grid(rotated, self.grid_n)

            agree = int(np.count_nonzero(cand_grid == tpl_grid))
            total = self.grid_n * self.grid_n
            conf  = agree / total

            if conf > best_conf:
                best_conf = conf
                best_rot  = k * 90

        logger.debug(
            f"[QR] match  best_conf={best_conf:.3f}  "
            f"best_rot={best_rot}°  threshold={threshold:.2f}")

        return best_conf >= threshold, round(best_conf, 4)

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy save helper
    # ──────────────────────────────────────────────────────────────────────────

    def save_qr_data(self,
                     data_string: str,
                     raw_crop:    Optional[np.ndarray]) -> bool:
        try:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.output_dir, f"qr_data_{ts}")
            os.makedirs(save_dir, exist_ok=True)
            self.qr_save_dir = save_dir

            preview = (data_string[:200] + "…[truncated]"
                       if len(data_string) > 200 else data_string)

            coords_file = os.path.join(save_dir, f"coords_{ts}.txt")
            with open(coords_file, 'w') as f:
                f.write(preview)
            logger.info(f"[QR] Saved coords → {coords_file}")

            if raw_crop is not None:
                img_file = os.path.join(save_dir, f"target_{ts}.png")
                cv2.imwrite(img_file, raw_crop)
                logger.info(f"[QR] Saved crop   → {img_file}")

            return True
        except Exception as e:
            logger.error(f"[QR] save_qr_data failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, detections: list) -> None:
        for entry in detections:
            if len(entry) == 5:
                data, (cx, cy), rect, raw_crop, binary_crop = entry
            else:
                data, (cx, cy), rect, raw_crop = entry
                binary_crop = None

            rx, ry, rw, rh = rect

            if binary_crop is not None and self._target_template is not None:
                found, conf = self.match_against_target(binary_crop)
                box_color   = (0, 255, 0) if found else (255, 0, 255)
                conf_text   = f"CONF:{conf:.2f}"
            else:
                box_color = (255, 0, 255)
                conf_text = ""

            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)
            cv2.circle(frame, (cx, cy), 5, box_color, -1)

            label = (data[:20] + "…") if len(data) > 20 else (data or "QR")
            cv2.putText(frame, f"QR: {label}",
                        (rx, max(ry - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            status = conf_text if conf_text else (
                "[CROP OK]" if raw_crop is not None else "[NO CROP]")
            cv2.putText(frame, status,
                        (rx, ry + rh + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _image_to_module_grid(binary: np.ndarray,
                               grid_n: int) -> np.ndarray:
        """
        Sample *binary* onto an NxN module grid.
        Returns uint8 grid where 0 = black module, 1 = white module.
        """
        h, w = binary.shape[:2]
        cell_h = h / grid_n
        cell_w = w / grid_n
        grid   = np.zeros((grid_n, grid_n), dtype=np.uint8)

        for i in range(grid_n):
            y1 = int(i       * cell_h)
            y2 = int((i + 1) * cell_h)
            for j in range(grid_n):
                x1 = int(j       * cell_w)
                x2 = int((j + 1) * cell_w)
                cell = binary[y1:y2, x1:x2]
                grid[i, j] = 0 if cell.mean() < 127 else 1
        return grid

    @staticmethod
    def _extract_base64(payload: str) -> Optional[str]:
        """
        Strip wrapper / data URI and return bare base64.
        Returns None if it doesn't decode to a recognised image.
        """
        s = payload.strip()

        if s.startswith("data:") and ";base64," in s:
            s = s.split(";base64,", 1)[1]

        for prefix in ("IMG:", "TARGET:", "TPL:", "B64:"):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break

        s = s.strip().replace("\n", "").replace("\r", "").replace(" ", "")

        if len(s) < 100:
            logger.debug(f"[QR] payload too short for image base64: {len(s)}")
            return None

        try:
            decoded = base64.b64decode(s, validate=False)
            if (decoded[:4] == b'\x89PNG'
                    or decoded[:3] == b'\xff\xd8\xff'   # JPEG
                    or decoded[:2] == b'BM'             # BMP
                    or decoded[:4] == b'GIF8'):         # GIF
                return s
            logger.debug(
                f"[QR] decoded bytes don't look like image: {decoded[:8]}")
            return None
        except Exception as e:
            logger.debug(f"[QR] base64 validation failed: {e}")
            return None

    def _crop_qr(self, frame: np.ndarray, rect) -> Optional[np.ndarray]:
        try:
            x, y, w, h = rect.left, rect.top, rect.width, rect.height
            margin = max(4, int(min(w, h) * 0.08))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            crop = frame[y1:y2, x1:x2]
            return crop.copy() if crop.size > 0 else None
        except Exception as e:
            logger.debug(f"[QR] _crop_qr: {e}")
            return None

    def _preprocess_template(self,
                              img:     np.ndarray,
                              denoise: bool = False
                              ) -> Optional[np.ndarray]:
        """
        Canonical QR → binary pipeline.

        CLEAN path (denoise=False):
            gray → Otsu → trim_quiet_zone → resize → re-binarise
        NOISY path (denoise=True):
            gray → blur → CLAHE → adaptiveThresh → trim_quiet_zone → resize → re-binarise
        """
        try:
            gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(img.shape) == 3 else img.copy())

            if denoise:
                gray   = cv2.GaussianBlur(gray, (3, 3), 0)
                eq     = self._clahe.apply(gray)
                binary = cv2.adaptiveThreshold(
                    eq, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11, 2)
            else:
                _, binary = cv2.threshold(
                    gray, 0, 255,
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
            inv    = cv2.bitwise_not(binary)
            coords = cv2.findNonZero(inv)
            if coords is None:
                return None
            x, y, w, h = cv2.boundingRect(coords)
            m  = 2
            x1 = max(0, x - m)
            y1 = max(0, y - m)
            x2 = min(binary.shape[1], x + w + m)
            y2 = min(binary.shape[0], y + h + m)
            return binary[y1:y2, x1:x2]
        except Exception as e:
            logger.debug(f"[QR] _trim_quiet_zone: {e}")
            return None

    def _write_template_metadata(self,
                                  timestamp: str,
                                  filename:  str,
                                  qr_data:   str) -> None:
        meta_path = os.path.join(self.template_dir, "metadata.log")
        try:
            with open(meta_path, 'a') as f:
                f.write(f"{timestamp},{filename},{qr_data}\n")
        except Exception as e:
            logger.warning(f"[QR] metadata write failed: {e}")
