import cv2
import numpy as np

class RedZoneDetector:
    def __init__(self, hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2, min_area):
        self.hsv_lower1 = np.array(hsv_lower1)
        self.hsv_upper1 = np.array(hsv_upper1)
        self.hsv_lower2 = np.array(hsv_lower2)
        self.hsv_upper2 = np.array(hsv_upper2)
        self.min_area = min_area
    
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # RGB input
        mask1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
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
            cv2.rectangle(frame, (cx-s,cy-s), (cx+s,cy+s), (0,0,255), 2)
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
            cv2.putText(frame, f"R:{int(area)}", (cx-s,cy-s-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
