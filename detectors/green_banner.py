# green_banner.py
import cv2
import numpy as np

class GreenBannerDetector:
    __slots__ = ('hsv_lower', 'hsv_upper', 'min_area', 'kernel')
    
    def __init__(self, hsv_lower, hsv_upper, min_area):
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = min_area
        self.kernel = np.ones((5,5), np.uint8)
    
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x,y,w,h = cv2.boundingRect(cnt)
                results.append(((x+w//2, y+h//2), area))
        return results
    
    def draw(self, frame, detections):
        for (cx,cy), area in detections:
            s = 20
            cv2.rectangle(frame, (cx-s,cy-s), (cx+s,cy+s), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)
            cv2.putText(frame, f"G:{int(area)}", (cx-s,cy-s-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)