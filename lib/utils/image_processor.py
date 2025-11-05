import cv2
import numpy as np
from typing import List

from ..results import DetectionResult

class ImageProcessor:
    """Utility cho xử lý ảnh"""
    
    def __init__(self):
        self.colors = {
            'plate': (0, 255, 0),     # Green
            'container': (255, 0, 0), # Blue  
            'face': (0, 0, 255),      # Red
        }
    
    def draw_detections(self, image: np.ndarray, 
                       detections: List[DetectionResult]) -> np.ndarray:
        """Vẽ bounding boxes và labels lên ảnh"""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.colors.get(det.detection_type, (255, 255, 255))
            
            # Vẽ bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label
            label = f"{det.detection_type}: {det.text or 'None'} ({det.confidence:.2f})"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def resize_image(self, image: np.ndarray, max_size: int = 1920) -> np.ndarray:
        """Resize ảnh để tối ưu performance"""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image