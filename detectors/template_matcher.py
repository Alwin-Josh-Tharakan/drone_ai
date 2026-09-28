"""
detectors/template_matcher.py
Adaptive Binary Template Verification — fixed pipeline (optimized fork for the
SINOSEE global-shutter USB camera).

Pipeline (candidate side)
-------------------------
Phase 1 — Illumination Normalization  (grayscale + CLAHE)
Phase 2 — Robust Binarization         (shared QRDetector.binarize_qr:
                                         CLAHE → Otsu, module-scaled adaptive
                                         fallback — keeps modules SOLID)
Phase 3 — Spatial Normalization       (contour → minAreaRect → de-rotate → resize)
Phase 4 — Similarity Score            (pixel similarity + NCC, best of 4 rotations)
Phase 5 — Confidence Thresholding     (emit MatchResult)
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .qr_detector import QRDetector

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Output of a single template match attempt."""
    match_found: bool = False
    confidence:  float = 0.0    # 0.0 – 1.0
    center:      tuple = (0, 0)  # (cx, cy) in full-frame coords
    angle:       float = 0.0    # detected rotation (degrees)
    deviation_x: int = 0        # px offset from frame centre
    deviation_y: int = 0        # px offset from frame centre


class TemplateMatcher:
    """
    Adaptive Binary Template Verification with Spatial Normalization.
    Outputs deviation_x/deviation_y consumed by the mission logic to align
    the drone before payload release.
    """

    def __init__(self,
                 match_threshold: float = 0.75,
                 template_size: tuple = (128, 128),
                 clahe_limit: float = 2.0,
                 clahe_tile_size: tuple = (8, 8),
                 frame_size: tuple = (1600, 1300)):  # SINOSEE USB resolution
        self.match_threshold = match_threshold
        self.template_size = template_size
        self.frame_center = (frame_size[0] // 2, frame_size[1] // 2)

        # Single CLAHE instance reused every frame
        self._clahe = cv2.createCLAHE(clipLimit=clahe_limit,
                                      tileGridSize=clahe_tile_size)

    # ──────────────────────────────────────────────────────────────────────
    # Shared preprocessing — Phase 1 + 2
    # ──────────────────────────────────────────────────────────────────────

    def process_to_binary(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Input : BGR or grayscale image (any size).
        Output: binary uint8 image, same size as input, values 0 or 255.
        Uses the SAME scale-aware binariser as the candidate pipeline so
        template and candidate are always compared apples-to-apples.
        """
        try:
            gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if img.ndim == 3 else img.copy())
            return QRDetector.binarize_qr(gray, self._clahe)
        except Exception as e:
            logger.error(f"[MATCHER] process_to_binary: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def preprocess_candidate(self, frame: np.ndarray,
                             rect: Optional[tuple] = None) -> Optional[np.ndarray]:
        """
        Crop the candidate QR region from the full frame (single 8% margin —
        identical to QRDetector._crop_qr). Perspective correction is NOT
        applied here; it happens inside _spatial_normalise() after
        binarisation, matching the template pipeline exactly.
        """
        try:
            if rect is not None:
                x, y, w, h = rect
                margin = max(4, int(min(w, h) * 0.08))
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                crop = frame[y1:y2, x1:x2].copy()
            else:
                crop = frame.copy()
            return crop if crop.size else None
        except Exception as e:
            logger.warning(f"[MATCHER] preprocess_candidate: {e}")
            return None

    def match(self,
              candidate_frame: np.ndarray,
              template_binary: np.ndarray,
              candidate_rect: Optional[tuple] = None) -> MatchResult:
        """
        Compare a live QR crop against the stored binary template.

        Parameters
        ----------
        candidate_frame : BGR/gray crop from preprocess_candidate().
        template_binary : TargetData.template_qr_image (binary, template_size).
        candidate_rect  : (x, y, w, h) in full frame — for deviation calc.
        """
        if template_binary is None or template_binary.size == 0:
            logger.warning("[MATCHER] No template loaded")
            return MatchResult()

        # Phase 1 + 2
        binary = self.process_to_binary(candidate_frame)
        if binary is None:
            return MatchResult()

        # Phase 3
        normalised, angle, center_local = self._spatial_normalise(binary)
        if normalised is None:
            logger.debug("[MATCHER] Spatial normalisation failed")
            return MatchResult()

        # Phase 4
        confidence = self._combined_similarity(normalised, template_binary)

        # Phase 5
        match_found = confidence >= self.match_threshold
        center, dx, dy = self._compute_deviation(
            candidate_rect, candidate_frame.shape, center_local)

        result = MatchResult(
            match_found=match_found,
            confidence=round(confidence, 4),
            center=center,
            angle=round(angle, 2),
            deviation_x=dx,
            deviation_y=dy,
        )

        if match_found:
            logger.info(f"[MATCHER] ✓ MATCH conf={confidence:.3f} angle={angle:.1f}°")
        else:
            logger.debug(f"[MATCHER] ✗ No match conf={confidence:.3f}")
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Internal — pipeline phases
    # ──────────────────────────────────────────────────────────────────────

    def _spatial_normalise(self, binary: np.ndarray
                           ) -> Tuple[Optional[np.ndarray], float, tuple]:
        """Phase 3: largest contour → minAreaRect → de-rotate → resize."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, (0, 0)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None, 0.0, (0, 0)

        rect_rot = cv2.minAreaRect(largest)
        center_local, (rw, rh), raw_angle = rect_rot

        # Fix minAreaRect angle ambiguity: keep result in (-45, +45]
        angle = self._correct_angle(raw_angle, rw, rh)

        M = cv2.getRotationMatrix2D(center_local, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (binary.shape[1], binary.shape[0]),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REPLICATE)

        box = cv2.boxPoints(rect_rot)
        box_rot = cv2.transform(box.reshape(-1, 1, 2).astype(np.float32),
                                M).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(box_rot.astype(np.int32))
        x, y = max(0, x), max(0, y)
        w = min(w, rotated.shape[1] - x)
        h = min(h, rotated.shape[0] - y)
        if w < 8 or h < 8:
            return None, angle, tuple(map(int, center_local))

        cropped = rotated[y:y + h, x:x + w]

        # Trim quiet zone so both template and candidate contain ONLY modules
        trimmed = QRDetector._trim_quiet_zone(cropped)
        if trimmed is None or trimmed.size == 0:
            trimmed = cropped

        resized = cv2.resize(trimmed, self.template_size,
                             interpolation=cv2.INTER_AREA)
        _, normalised = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        return normalised, angle, tuple(map(int, center_local))

    @staticmethod
    def _correct_angle(angle: float, w: float, h: float) -> float:
        """
        cv2.minAreaRect returns angles in [-90, 0). If the box is wider than
        tall the meaningful rotation is angle+90. Clamp result to (-45, +45].
        """
        if w < h:
            angle += 90.0
        while angle > 45.0:
            angle -= 90.0
        while angle <= -45.0:
            angle += 90.0
        return angle

    def _combined_similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Phase 4: pixel similarity (absdiff) + NCC, averaged.
        Best score over rot90^k — fixes 90°-rotated candidates."""
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            _, img_b = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)

        best = 0.0
        a = img_a
        for k in range(4):
            if k > 0:
                a = np.rot90(a)
            mismatch = np.count_nonzero(cv2.absdiff(a, img_b))
            pixel_sim = 1.0 - mismatch / a.size
            ncc = max(0.0, float(cv2.matchTemplate(a, img_b,
                                                   cv2.TM_CCOEFF_NORMED)[0, 0]))
            best = max(best, (pixel_sim + ncc) / 2.0)

        logger.debug(f"[MATCHER] combined={best:.3f}")
        return round(max(0.0, min(1.0, best)), 4)

    def _compute_deviation(self, rect, frame_shape,
                           center_local) -> Tuple[tuple, int, int]:
        """Return absolute (cx, cy) in full-frame coords and (dx, dy)."""
        fc_x, fc_y = self.frame_center
        if rect is not None:
            rx, ry, rw, rh = rect
            cx, cy = rx + rw // 2, ry + rh // 2
        else:
            cx, cy = int(center_local[0]), int(center_local[1])
        return (cx, cy), cx - fc_x, cy - fc_y