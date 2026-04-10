from pyzbar.pyzbar import decode
import cv2

class QRDetector:
    def __init__(self, min_size=50):
        self.min_size = min_size
    
    def detect(self, frame):
        """Returns [(data, center)]"""
        results = []
        decoded = decode(frame)
        
        for obj in decoded:
            data = obj.data.decode('utf-8')
            rect = obj.rect
            center = (rect.left + rect.width//2, rect.top + rect.height//2)
            
            if rect.width >= self.min_size and rect.height >= self.min_size:
                results.append((data, center))  # Just data and center
        
        return results
    
    def draw(self, frame, detections):
        for data, (cx, cy) in detections:
            # Draw box around center
            size = 30
            cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (255,0,255), 2)
            cv2.circle(frame, (cx, cy), 5, (255,0,255), -1)
            cv2.putText(frame, f"QR: {data}", (cx-size, cy-size-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
