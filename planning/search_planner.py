# planning/search_planner.py
"""
Search-path planning module.

Strategies
----------
* BoustrophedonPlanner — lawnmower sweep, starts at bottom-LEFT and ends at
  bottom-RIGHT (or similar configurable corner).  Waypoints are returned in
  NORMALISED coordinates (0.0–1.0) where:
        (0, 0) = bottom-left of plot
        (1, 1) = top-right of plot

* NullPlanner — fallback that does nothing.  Used automatically if the real
  planner fails to import or crashes during construction.

The vision_combined script must NEVER crash because of the planner —
all calls into the planner are wrapped in try/except and any failure
silently degrades to the NullPlanner.

Public API (any planner must implement these)
---------------------------------------------
    plan(rows: int) -> List[Tuple[float, float]]
    advance() -> Optional[Tuple[float, float]]
    current_waypoint() -> Optional[Tuple[float, float]]
    remaining() -> int
    reset() -> None
    is_complete() -> bool
"""

from __future__ import annotations
import json
import time
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Base interface
# ──────────────────────────────────────────────────────────────────────────────

class BasePlanner:
    """Common interface — every planner subclass must respect these signatures."""

    name: str = "base"

    def plan(self, rows: int) -> List[Tuple[float, float]]:
        raise NotImplementedError

    def advance(self) -> Optional[Tuple[float, float]]:
        raise NotImplementedError

    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        raise NotImplementedError

    def remaining(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def is_complete(self) -> bool:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Null planner — fail-safe fallback that never crashes
# ──────────────────────────────────────────────────────────────────────────────

class NullPlanner(BasePlanner):
    """Does nothing — used when planning is disabled or has crashed."""

    name = "null"

    def __init__(self):
        logger.warning(
            "[PLANNER] NullPlanner active — search planning is DISABLED")

    def plan(self, rows: int = 0) -> List[Tuple[float, float]]:
        return []

    def advance(self) -> Optional[Tuple[float, float]]:
        return None

    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        return None

    def remaining(self) -> int:
        return 0

    def reset(self) -> None:
        pass

    def is_complete(self) -> bool:
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Boustrophedon (lawnmower) planner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BoustrophedonPlanner(BasePlanner):
    """
    Lawnmower / boustrophedon coverage in NORMALISED coordinates.

    Default sweep
    -------------
    Starts at bottom-LEFT (0, 0) and ends at bottom-RIGHT (1, 0) on the
    final row.  The number of rows is configurable.

    Path shape (rows=4):
        row 0:   (0,0)→(1,0)
        row 1:   (1,t)→(0,t)
        row 2:   (0,2t)→(1,2t)
        row 3:   (1,3t)→(0,3t)
    where t = 1/(rows-1).

    Then a final segment returns to bottom-right (1, 0) so the path ends
    where you asked it to end.
    """

    rows:                 int   = 8
    hold_seconds:         float = 1.0
    waypoints:            List[Tuple[float, float]] = field(default_factory=list)
    _index:               int   = 0
    _last_advance_time:   float = field(default_factory=lambda: time.time())
    name:                 str   = "boustrophedon"

    # ── Plan generation ──────────────────────────────────────────────────────

    def plan(self, rows: Optional[int] = None) -> List[Tuple[float, float]]:
        if rows is not None and rows > 0:
            self.rows = rows
        if self.rows < 2:
            self.rows = 2

        path: List[Tuple[float, float]] = []
        t = 1.0 / (self.rows - 1)

        for r in range(self.rows):
            y = r * t
            if r % 2 == 0:                       # even row → left → right
                path.append((0.0, y))
                path.append((1.0, y))
            else:                                # odd row  → right → left
                path.append((1.0, y))
                path.append((0.0, y))

        # Final descent / return to bottom-right (user's requested end point)
        if path and path[-1] != (1.0, 0.0):
            path.append((1.0, 0.0))

        self.waypoints         = path
        self._index            = 0
        self._last_advance_time = time.time()

        logger.info(
            f"[PLANNER] Boustrophedon plan built  "
            f"rows={self.rows}  waypoints={len(path)}")
        return path

    # ── Iteration ────────────────────────────────────────────────────────────

    def advance(self) -> Optional[Tuple[float, float]]:
        """
        Move to next waypoint IF hold_seconds has elapsed since the last move.
        Returns the new current waypoint, or None if path complete.
        """
        if self.is_complete():
            return None

        now = time.time()
        if now - self._last_advance_time >= self.hold_seconds:
            self._index += 1
            self._last_advance_time = now

        return self.current_waypoint()

    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        if 0 <= self._index < len(self.waypoints):
            return self.waypoints[self._index]
        return None

    def remaining(self) -> int:
        return max(0, len(self.waypoints) - self._index - 1)

    def reset(self) -> None:
        self._index = 0
        self._last_advance_time = time.time()

    def is_complete(self) -> bool:
        return self._index >= len(self.waypoints)


# ──────────────────────────────────────────────────────────────────────────────
# Factory + crash-safe builder
# ──────────────────────────────────────────────────────────────────────────────

def build_planner(strategy: str  = "boustrophedon",
                  rows:     int  = 8,
                  hold_sec: float = 1.0) -> BasePlanner:
    """
    Build a planner by name.  ALWAYS returns a usable object — falls back to
    NullPlanner on any exception.
    """
    try:
        if strategy == "boustrophedon":
            p = BoustrophedonPlanner(rows=rows, hold_seconds=hold_sec)
            p.plan()
            return p
        else:
            logger.warning(
                f"[PLANNER] Unknown strategy '{strategy}' — using NullPlanner")
            return NullPlanner()
    except Exception as e:
        logger.exception(f"[PLANNER] build failed ({e}) — using NullPlanner")
        return NullPlanner()


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers — never crash, return frame unchanged on error
# ──────────────────────────────────────────────────────────────────────────────

def _norm_to_pixel(pt: Tuple[float, float],
                   width: int,
                   height: int,
                   margin: int) -> Tuple[int, int]:
    """
    Map normalised (x, y) ∈ [0,1] to frame pixels.
    (0,0) = bottom-LEFT of usable area
    (1,1) = top-RIGHT  of usable area
    """
    x_norm, y_norm = pt
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))

    px = int(margin + x_norm * (width  - 2 * margin))
    # invert y because pixel-y grows downwards
    py = int((height - margin) - y_norm * (height - 2 * margin))
    return px, py


def draw_path(frame: np.ndarray,
              planner: BasePlanner,
              margin: int                = 30,
              color_pending: Tuple[int, int, int] = (180, 180, 180),
              color_visited: Tuple[int, int, int] = ( 80,  80, 200),
              color_current: Tuple[int, int, int] = (  0, 255,   0),
              color_path:    Tuple[int, int, int] = (  0, 200, 255)) -> None:
    """
    Draw the planner's path on *frame* in place.

    NEVER raises — any error is logged and the frame is left unchanged.
    """
    try:
        wpts = getattr(planner, "waypoints", None)
        if not wpts:
            return

        h, w = frame.shape[:2]
        cur_idx = getattr(planner, "_index", 0)

        # Draw plot boundary (the unit square)
        tl = _norm_to_pixel((0.0, 1.0), w, h, margin)
        br = _norm_to_pixel((1.0, 0.0), w, h, margin)
        cv2.rectangle(frame, tl, br, color_path, 1, cv2.LINE_AA)

        # Draw path segments
        pixel_pts = [_norm_to_pixel(p, w, h, margin) for p in wpts]
        for i in range(len(pixel_pts) - 1):
            c = color_visited if i < cur_idx else color_pending
            cv2.line(frame, pixel_pts[i], pixel_pts[i + 1], c, 1, cv2.LINE_AA)

        # Draw individual waypoint markers
        for i, p in enumerate(pixel_pts):
            if i < cur_idx:
                cv2.circle(frame, p, 3, color_visited, -1)
            elif i == cur_idx:
                cv2.circle(frame, p, 6, color_current, -1)
                cv2.circle(frame, p, 9, color_current,  2)
            else:
                cv2.circle(frame, p, 2, color_pending, -1)

        # Label start + end
        cv2.putText(frame, "S", pixel_pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "E", pixel_pts[-1],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Status text
        total = len(wpts)
        done  = min(cur_idx, total)
        cv2.putText(
            frame,
            f"PLAN: {planner.name}  {done}/{total}",
            (margin, margin - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_path, 1)

    except Exception as e:
        logger.debug(f"[PLANNER] draw_path skipped: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_plan(planner: BasePlanner, out_dir: str) -> None:
    """
    Persist waypoints + a preview PNG of the path.  Never raises.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        wpts = getattr(planner, "waypoints", [])
        if not wpts:
            return

        # JSON waypoints
        json_path = os.path.join(out_dir, "waypoints.json")
        with open(json_path, "w") as f:
            json.dump({
                "strategy":  planner.name,
                "count":     len(wpts),
                "waypoints": [[float(x), float(y)] for x, y in wpts],
            }, f, indent=2)
        logger.info(f"[PLANNER] Saved waypoints → {json_path}")

        # Preview PNG
        preview = np.full((480, 640, 3), 30, dtype=np.uint8)
        class _Tmp:
            waypoints = wpts
            _index = 0
            name = planner.name
        draw_path(preview, _Tmp(), margin=30)
        prev_path = os.path.join(out_dir, "path_preview.png")
        cv2.imwrite(prev_path, preview)
        logger.info(f"[PLANNER] Saved preview   → {prev_path}")
    except Exception as e:
        logger.debug(f"[PLANNER] save_plan skipped: {e}")