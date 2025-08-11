from typing import List, Tuple
import importlib.resources as pkg_resources
import numpy as np
from ultralytics import YOLO

from ..results import DetectionResult, ContainerResult
from .base_detector import BaseDetector
from ..ocr import ContainerOCR

class ContainerDetector(BaseDetector):
    """Detector cho biến số xe"""
    def _load_model(self):
        with pkg_resources.path('lib.model', 'detect_ContainerNumber.pt') as model_path:
            self.model = YOLO(str(model_path)).to(self.device)
        self.ocr = ContainerOCR()

    def _safe_crop(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 10) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        h, w = image.shape[:2]
        x1_c = max(0, x1 - pad); y1_c = max(0, y1 - pad)
        x2_c = min(w, x2 + pad); y2_c = min(h, y2 + pad)
        if x2_c <= x1_c or y2_c <= y1_c:
            return image[0:0, 0:0], (x1, y1, x2, y2)
        return image[y1_c:y2_c, x1_c:x2_c], (x1, y1, x2, y2)

    def detect(self, image: np.ndarray) -> List[DetectionResult]: # type: ignore
        """
        Phát hiện mã container trong ảnh
        Args:
            image: Ảnh đầu vào (BGR format)
        Returns:
            List[DetectionResult]: Danh sách mã container phát hiện được
        """

        results: List[DetectionResult] = []
        try:
            y = self.model(image, conf=0.35, iou=0.5)
            for box in y.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy)
                cropped, bbox = self._safe_crop(image, x1, y1, x2, y2, pad=10)
                if cropped.size == 0:
                    results.append(DetectionResult('container', [x1,y1,x2,y2], conf, None))
                    continue
                try:
                    text, ocr_conf, orientation = self.ocr.extract_text(cropped)  # type: ignore
                except Exception as e:
                    print(f"[ERROR] Container OCR error: {e}")
                    text, ocr_conf, orientation = None, 0.0, "horizontal"
                results.append(ContainerResult(
                    detection_type='container',
                    bbox=[bbox, bbox, bbox, bbox], # type: ignore
                    confidence=float(ocr_conf if text else conf),
                    text=text,
                    detection_confidence=conf,
                    ocr_confidence=float(ocr_conf if text else 0.0),
                    orientation=orientation
                ))
        except Exception as e:
            print(f"[ERROR] Container detection failed: {e}")
        return results

