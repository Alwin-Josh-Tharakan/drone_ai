# qr_detector.py
from pyzbar.pyzbar import decode
import cv2

class QRDetector:
    __slots__ = ('min_size',)
    
    def __init__(self, min_size=50):
        self.min_size = min_size
    
    def detect(self, frame):
        decoded = decode(frame)
        return [(obj.data.decode('utf-8'), 
                (obj.rect.left + obj.rect.width//2, obj.rect.top + obj.rect.height//2))
                for obj in decoded if obj.rect.width >= self.min_size]
    
    def draw(self, frame, detections):
        for data, (cx, cy) in detections:
            size = 30
            cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (255,0,255), 2)
            cv2.circle(frame, (cx, cy), 5, (255,0,255), -1)
            cv2.putText(frame, f"QR: {data}", (cx-size, cy-size-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)