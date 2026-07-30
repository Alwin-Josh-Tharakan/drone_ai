# detectors/red_zone.py
# Feature 4: structured red zone detection with solidity / aspect-ratio filters
# Bug fix: draw() previously expected (center, area) but detect() now returns
#          (cx, cy, area) — draw() updated to match.

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RedZoneDetector:
    """
    Detects structured red zones (5 m × 5 m or 10 m × 10 m ground markings).

    Filters applied beyond simple area threshold
    ────────────────────────────────────────────
    • min_area         — eliminates small random red objects
    • min_solidity     — (area / convex-hull area) — rejects thin lines,
                         irregular blobs, vegetation edges
    • min_aspect_ratio — (shorter / longer side) — rejects narrow strips
    """

    def __init__(self,
                 hsv_lower1, hsv_upper1,
                 hsv_lower2, hsv_upper2,
                 min_area:         int   = 5000,
                 min_solidity:     float = 0.70,
                 min_aspect_ratio: float = 0.40):
        self.hsv_lower1       = np.array(hsv_lower1)
        self.hsv_upper1       = np.array(hsv_upper1)
        self.hsv_lower2       = np.array(hsv_lower2)
        self.hsv_upper2       = np.array(hsv_upper2)
        self.min_area         = min_area
        self.min_solidity     = min_solidity
        self.min_aspect_ratio = min_aspect_ratio

        # 7×7 kernel for strict morphological cleanup
        self._kernel = np.ones((7, 7), np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list:
        """
        Returns [(cx, cy, area), ...] sorted by area descending.
        Only STRUCTURED (large, solid, roughly-square) red zones pass.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Dual-range mask (red wraps around hue=0/180)
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup — remove noise and fill internal gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue  # too small

            # ── Solidity filter ──────────────────────────────
            hull     = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < self.min_solidity:
                logger.debug(f"[RED] Rejected: solidity={solidity:.2f} < {self.min_solidity}")
                continue  # irregular / ragged blob

            # ── Aspect-ratio filter ──────────────────────────
            _, (rw, rh), _ = cv2.minAreaRect(cnt)
            if rw == 0 or rh == 0:
                continue
            aspect = min(rw, rh) / max(rw, rh)
            if aspect < self.min_aspect_ratio:
                logger.debug(f"[RED] Rejected: aspect={aspect:.2f} < {self.min_aspect_ratio}")
                continue  # thin line / narrow strip

            # ── Centroid ─────────────────────────────────────
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            results.append((cx, cy, area))

        # Largest zone first — most dangerous
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def draw(self, frame: np.ndarray, detections: list) -> None:
        """Draw red-zone warning boxes. detections = [(cx, cy, area), ...]"""
        for (cx, cy, area) in detections:
            size = int(np.sqrt(area) / 4)
            cv2.rectangle(frame,
                          (cx - size, cy - size),
                          (cx + size, cy + size),
                          (0, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"DANGER {int(area)}",
                        (cx - size, cy - size - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, "RED ZONE",
                        (cx - size, cy - size - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)