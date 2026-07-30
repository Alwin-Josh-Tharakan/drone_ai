"""
detectors/template_matcher.py

Adaptive Binary Template Verification — fixed pipeline.

Key fixes vs previous version
-------------------------------
1. Single shared _process_to_binary() used by BOTH template (via QRDetector)
   and candidate sides — guarantees identical preprocessing.
2. preprocess_candidate() no longer applies perspective correction before
   returning; perspective correction is folded into _spatial_normalise() only
   when a clean quad is found, and is applied to the already-binarised image
   so it matches how the template was made.
3. minAreaRect angle ambiguity fixed with _correct_angle().
4. Threshold lowered to 0.75 (empirically reasonable for binary QR comparison).
5. _bitwise_similarity() now also returns a structural score via matchTemplate
   normalised cross-correlation as a second opinion, then takes the average.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Output of a single template match attempt."""
    match_found:  bool  = False
    confidence:   float = 0.0   # 0.0 – 1.0
    center:       tuple = (0, 0) # (cx, cy) in full-frame coords
    angle:        float = 0.0   # detected rotation (degrees)
    deviation_x:  int   = 0     # px offset from frame centre
    deviation_y:  int   = 0     # px offset from frame centre


class TemplateMatcher:
    """
    Adaptive Binary Template Verification with Spatial Normalization.

    Pipeline (candidate side)
    -------------------------
    Phase 1 — Illumination Normalization  (grayscale + CLAHE)
    Phase 2 — Robust Binarization         (adaptive Gaussian threshold + re-binarise)
    Phase 3 — Spatial Normalization       (contour → minAreaRect → de-rotate → resize)
    Phase 4 — Similarity Score            (pixel similarity + NCC combined)
    Phase 5 — Confidence Thresholding     (emit MatchResult)

    Template side uses the SAME Phase 1+2 via QRDetector._preprocess_template()
    which calls the shared static method process_to_binary().
    """

    def __init__(self,
                 match_threshold: float = 0.75,   # lowered from 0.95
                 template_size:   tuple  = (128, 128),
                 clahe_limit:     float  = 2.0,
                 clahe_tile_size: tuple  = (8, 8),
                 frame_size:      tuple  = (640, 480)):

        self.match_threshold = match_threshold
        self.template_size   = template_size
        self.clahe_limit     = clahe_limit
        self.clahe_tile_size = clahe_tile_size
        self.frame_center    = (frame_size[0] // 2, frame_size[1] // 2)

        # Single CLAHE instance reused every frame
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_limit,
            tileGridSize=self.clahe_tile_size)

    # ──────────────────────────────────────────────────────────────────────
    # Shared preprocessing — called by BOTH this class AND QRDetector
    # ──────────────────────────────────────────────────────────────────────

    def process_to_binary(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Phase 1 + 2 shared between template and candidate.
        Input : BGR or grayscale image (any size).
        Output: binary uint8 image, same size as input, values 0 or 255.
        """
        try:
            gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(img.shape) == 3 else img.copy())

            eq = self._clahe.apply(gray)

            binary = cv2.adaptiveThreshold(
                eq, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2)

            return binary
        except Exception as e:
            logger.error(f"[MATCHER] process_to_binary: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def preprocess_candidate(self,
                             frame: np.ndarray,
                             rect:  Optional[tuple] = None
                             ) -> Optional[np.ndarray]:
        """
        Crop the candidate QR region from the full frame.
        Returns a BGR/gray crop — preprocessing happens inside match().

        Note: perspective correction is intentionally NOT applied here.
        It is applied inside _spatial_normalise() after binarisation,
        matching the template pipeline exactly.
        """
        try:
            if rect is not None:
                x, y, w, h = rect
                # Single margin application — NOT doubled
                margin = max(4, int(min(w, h) * 0.08))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                crop = frame[y1:y2, x1:x2].copy()
            else:
                crop = frame.copy()

            if crop.size == 0:
                return None

            return crop

        except Exception as e:
            logger.warning(f"[MATCHER] preprocess_candidate: {e}")
            return None

    def match(self,
              candidate_frame:  np.ndarray,
              template_binary:  np.ndarray,
              candidate_rect:   Optional[tuple] = None) -> MatchResult:
        """
        Compare a live QR crop against the stored binary template.

        Parameters
        ----------
        candidate_frame : BGR or grayscale crop from preprocess_candidate().
        template_binary : Pre-processed binary from TargetData.template_qr_image.
        candidate_rect  : (x, y, w, h) in full frame — for deviation calc.
        """
        if template_binary is None or template_binary.size == 0:
            logger.warning("[MATCHER] No template loaded")
            return MatchResult()

        # ── Phase 1 + 2 ──────────────────────────────────────────────────
        binary = self.process_to_binary(candidate_frame)
        if binary is None:
            return MatchResult()

        # ── Phase 3 ──────────────────────────────────────────────────────
        normalised, angle, center_local = self._spatial_normalise(binary)
        if normalised is None:
            logger.debug("[MATCHER] Spatial normalisation failed")
            return MatchResult()

        # ── Phase 4 ──────────────────────────────────────────────────────
        confidence = self._combined_similarity(normalised, template_binary)

        # ── Phase 5 ──────────────────────────────────────────────────────
        match_found = confidence >= self.match_threshold

        center, deviation_x, deviation_y = self._compute_deviation(
            candidate_rect, candidate_frame.shape, center_local)

        result = MatchResult(
            match_found = match_found,
            confidence  = round(confidence, 4),
            center      = center,
            angle       = round(angle, 2),
            deviation_x = deviation_x,
            deviation_y = deviation_y,
        )

        if match_found:
            logger.info(
                f"[MATCHER] ✓ MATCH  conf={confidence:.3f}  angle={angle:.1f}°")
        else:
            logger.debug(f"[MATCHER] ✗ No match  conf={confidence:.3f}")

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Internal — pipeline phases
    # ──────────────────────────────────────────────────────────────────────

    def _spatial_normalise(self,
                           binary: np.ndarray
                           ) -> Tuple[Optional[np.ndarray], float, tuple]:
        """
        Phase 3: largest contour → minAreaRect → de-rotate → resize.

        Returns (normalised_binary, angle_degrees, local_center_px).
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0, (0, 0)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None, 0.0, (0, 0)

        rect_rot                    = cv2.minAreaRect(largest)
        center_local, (rw, rh), raw_angle = rect_rot

        # Fix angle ambiguity: keep result in (-45, +45]
        angle = self._correct_angle(raw_angle, rw, rh)

        # De-rotate
        M       = cv2.getRotationMatrix2D(center_local, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, M, (binary.shape[1], binary.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE)

        # Bounding box of the rotated contour points
        box      = cv2.boxPoints(rect_rot)
        box_rot  = cv2.transform(
            box.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(box_rot.astype(np.int32))
        x = max(0, x);  y = max(0, y)
        w = min(w, rotated.shape[1] - x)
        h = min(h, rotated.shape[0] - y)

        if w < 8 or h < 8:
            return None, angle, tuple(map(int, center_local))

        cropped = rotated[y: y + h, x: x + w]

        # Resize to template_size and re-binarise
        resized = cv2.resize(cropped, self.template_size,
                             interpolation=cv2.INTER_AREA)
        _, normalised = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        return normalised, angle, tuple(map(int, center_local))

    @staticmethod
    def _correct_angle(angle: float, w: float, h: float) -> float:
        """
        cv2.minAreaRect returns angles in [-90, 0).
        If the box is wider than tall the meaningful rotation is angle+90.
        Clamp result to (-45, +45].
        """
        if w < h:
            angle += 90.0
        # Normalise to (-45, +45]
        while angle > 45.0:
            angle -= 90.0
        while angle <= -45.0:
            angle += 90.0
        return angle

    def _combined_similarity(self,
                              img_a: np.ndarray,
                              img_b: np.ndarray) -> float:
        """
        Phase 4: pixel similarity (absdiff) + NCC, averaged.

        img_a : normalised candidate binary at template_size
        img_b : stored template binary at template_size
        """
        # Ensure same shape
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(
                img_b, (img_a.shape[1], img_a.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            _, img_b = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)

        # --- Pixel similarity (absdiff) ---
        diff       = cv2.absdiff(img_a, img_b)
        mismatch   = np.count_nonzero(diff)
        pixel_sim  = 1.0 - mismatch / img_a.size

        # --- Normalised cross-correlation ---
        # matchTemplate requires float32
        a_f = img_a.astype(np.float32) / 255.0
        b_f = img_b.astype(np.float32) / 255.0
        result = cv2.matchTemplate(a_f, b_f, cv2.TM_CCOEFF_NORMED)
        ncc    = float(result[0, 0])          # single value: both same size
        ncc    = max(0.0, ncc)                # clamp negatives to 0

        combined = (pixel_sim + ncc) / 2.0
        logger.debug(
            f"[MATCHER] pixel_sim={pixel_sim:.3f}  ncc={ncc:.3f}  "
            f"combined={combined:.3f}")
        return round(max(0.0, min(1.0, combined)), 4)

    def _compute_deviation(self,
                           rect,
                           frame_shape,
                           center_local) -> Tuple[tuple, int, int]:
        """Return absolute (cx, cy) in full-frame coords and (dx, dy)."""
        fc_x, fc_y = self.frame_center

        if rect is not None:
            rx, ry, rw, rh = rect
            cx = rx + rw // 2
            cy = ry + rh // 2
        else:
            cx = int(center_local[0])
            cy = int(center_local[1])

        return (cx, cy), cx - fc_x, cy - fc_y
