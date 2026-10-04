# detectors/corridor_nav.py — unchanged from original
import cv2
import numpy as np

class CorridorNavigator:
    __slots__ = ('edge_threshold', 'center_tolerance')

    def __init__(self, edge_threshold, center_tolerance):
        self.edge_threshold  = edge_threshold
        self.center_tolerance = center_tolerance

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w  = frame.shape[:2]
        roi   = edges[h // 3:2 * h // 3, :]
        left_edges  = np.where(roi[:, :w // 2].any(axis=0))[0]
        right_edges = np.where(roi[:, w // 2:].any(axis=0))[0] + w // 2

        if len(left_edges) > 0 and len(right_edges) > 0:
            center_line = (left_edges[-1] + right_edges[0]) // 2
            return (center_line, center_line - w // 2)
        return (w // 2, 0)

    def draw(self, frame, center_line, deviation):
        if center_line is None:
            return
        h, w = frame.shape[:2]
        # Corridor edge lines + center line
        cv2.line(frame, (center_line, 0), (center_line, h), (255, 255, 0), 2)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
        # Deviation arrow from frame center to corridor center (mid-height)
        cy_arrow = h // 2
        cv2.arrowedLine(frame, (w // 2, cy_arrow), (center_line, cy_arrow),
                        (255, 255, 0), 2, tipLength=0.15)
        status = "C" if abs(deviation) < self.center_tolerance else (
            "R" if deviation > 0 else "L")
        cv2.putText(frame, f"{status}:{deviation:+d}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)