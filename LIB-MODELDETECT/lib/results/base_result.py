from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass  
class DetectionResult:
    """Kết quả phát hiện cơ bản"""
    detection_type: str  # 'plate', 'container', 'face'
    bbox: List[int]      # [x1, y1, x2, y2]
    confidence: float    # Độ tin cậy tổng thể
    text: Optional[str] = None  # Text nhận dạng được
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.detection_type,
            'text': self.text,
            'bounding_box': self.bbox,
            'confidence': self.confidence
        }
    
    def to_text_summary(self) -> str:
        """Trả về tóm tắt dạng text"""
        return f"{self.detection_type.upper()}: {self.text or 'None'} | Box: {self.bbox} | Conf: {self.confidence:.3f}"

@dataclass  
class PlateResult(DetectionResult):
    """Kết quả phát hiện biển số xe"""
    detection_confidence: float = 0.0  # Confidence của YOLO detection
    ocr_confidence: float = 0.0       # Confidence của OCR
    is_multiline: bool = False        # Biển số 2 dòng hay không
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'detection_confidence': self.detection_confidence,
            'ocr_confidence': self.ocr_confidence,
            'is_multiline': self.is_multiline
        })
        return result

@dataclass
class ContainerResult(DetectionResult):
    """Kết quả phát hiện container code"""
    detection_confidence: float = 0.0
    ocr_confidence: float = 0.0
    orientation: str = "horizontal"  # horizontal/vertical
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'detection_confidence': self.detection_confidence,
            'ocr_confidence': self.ocr_confidence,
            'orientation': self.orientation
        })
        return result

@dataclass
class FaceResult(DetectionResult):
    """Kết quả phát hiện khuôn mặt"""
    detection_confidence: float = 0.0    # MTCNN confidence
    recognition_confidence: float = 0.0  # Face recognition confidence
    person_name: Optional[str] = None    # Tên người được nhận dạng
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            'detection_confidence': self.detection_confidence,
            'recognition_confidence': self.recognition_confidence,
            'person_name': self.person_name or self.text
        })
        return result
