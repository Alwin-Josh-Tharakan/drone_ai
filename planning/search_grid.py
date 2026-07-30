# planning/search_grid.py
# Feature 5: Search Grid Framework — framework only, no search logic yet.
# Provides a 2-D occupancy grid with cell-state tracking and a visualiser.

import numpy as np
import cv2
import logging
from enum import IntEnum
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CellState(IntEnum):
    UNKNOWN      = 0
    VISITED      = 1
    TARGET_FOUND = 2
    RED_ZONE     = 3
    OBSTACLE     = 4


# BGR palette for visualisation
_CELL_COLORS = {
    CellState.UNKNOWN:      (60,  60,  60),
    CellState.VISITED:      (80, 160,  80),
    CellState.TARGET_FOUND: (0,  255, 255),
    CellState.RED_ZONE:     (0,    0, 200),
    CellState.OBSTACLE:     (30,  30, 200),
}

_CELL_LABELS = {
    CellState.UNKNOWN:      "?",
    CellState.VISITED:      "V",
    CellState.TARGET_FOUND: "T",
    CellState.RED_ZONE:     "R",
    CellState.OBSTACLE:     "X",
}


class SearchGrid:
    """
    2-D occupancy grid for mission area tracking.

    Parameters
    ----------
    cols, rows : grid dimensions
    origin     : (lat, lon) of grid (0,0) corner — future GPS integration hook
    cell_meters: real-world size of each cell in metres
    """

    def __init__(self,
                 cols:         int   = 10,
                 rows:         int   = 10,
                 origin:       tuple = (0.0, 0.0),
                 cell_meters:  float = 5.0):
        self.cols        = cols
        self.rows        = rows
        self.origin      = origin
        self.cell_meters = cell_meters

        # Internal grid — dtype uint8 maps to CellState values
        self._grid = np.full((rows, cols), CellState.UNKNOWN, dtype=np.uint8)

        # Future integration hook — waypoint queue
        self._planned_path: list = []

    # ──────────────────────────────────────────────────────────────────────────
    # State mutation
    # ──────────────────────────────────────────────────────────────────────────

    def mark_visited(self, col: int, row: int) -> None:
        self._set(col, row, CellState.VISITED)

    def mark_target(self, col: int, row: int) -> None:
        self._set(col, row, CellState.TARGET_FOUND)
        logger.info(f"[GRID] Target found at ({col}, {row})")

    def mark_redzone(self, col: int, row: int) -> None:
        self._set(col, row, CellState.RED_ZONE)
        logger.warning(f"[GRID] Red zone at ({col}, {row})")

    def mark_obstacle(self, col: int, row: int) -> None:
        self._set(col, row, CellState.OBSTACLE)

    def get_state(self, col: int, row: int) -> CellState:
        if self._in_bounds(col, row):
            return CellState(self._grid[row, col])
        return CellState.UNKNOWN

    def reset(self) -> None:
        self._grid[:] = CellState.UNKNOWN
        self._planned_path.clear()
        logger.info("[GRID] Reset")

    # ──────────────────────────────────────────────────────────────────────────
    # Future integration hooks
    # ──────────────────────────────────────────────────────────────────────────

    def set_planned_path(self, waypoints: list) -> None:
        """Hook for search_strategy.py to inject a waypoint sequence."""
        self._planned_path = list(waypoints)

    def next_waypoint(self) -> Optional[Tuple[int, int]]:
        """Returns (col, row) of next unvisited planned cell, or None."""
        for col, row in self._planned_path:
            if self.get_state(col, row) == CellState.UNKNOWN:
                return (col, row)
        return None

    def gps_to_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Future GPS integration hook.
        Convert WGS-84 lat/lon to grid (col, row).
        Currently returns (0, 0) — implement with real projection when ready.
        """
        # TODO: implement haversine / UTM projection
        return (0, 0)

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def render(self, cell_px: int = 40,
               drone_cell: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Returns a BGR image of the grid for overlaying on the debug view.

        Parameters
        ----------
        cell_px    : pixel size of each cell
        drone_cell : (col, row) to draw the drone marker, or None
        """
        h = self.rows * cell_px
        w = self.cols * cell_px
        img = np.zeros((h, w, 3), dtype=np.uint8)

        for row in range(self.rows):
            for col in range(self.cols):
                state  = CellState(self._grid[row, col])
                color  = _CELL_COLORS[state]
                label  = _CELL_LABELS[state]
                x0, y0 = col * cell_px, row * cell_px
                x1, y1 = x0 + cell_px - 1, y0 + cell_px - 1

                cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)

                if cell_px >= 20:
                    cv2.putText(img, label,
                                (x0 + 4, y0 + cell_px - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (255, 255, 255), 1)

        # Drone marker
        if drone_cell is not None:
            dc, dr = drone_cell
            if self._in_bounds(dc, dr):
                cx = dc * cell_px + cell_px // 2
                cy = dr * cell_px + cell_px // 2
                cv2.circle(img, (cx, cy), cell_px // 3, (0, 200, 255), -1)
                cv2.circle(img, (cx, cy), cell_px // 3, (255, 255, 255), 1)

        return img

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _set(self, col: int, row: int, state: CellState) -> None:
        if self._in_bounds(col, row):
            self._grid[row, col] = state
        else:
            logger.warning(f"[GRID] Out of bounds: ({col}, {row})")

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows