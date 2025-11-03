from typing import List, Tuple
from importlib import resources
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from ..results import DetectionResult, PlateResult
from .base_detector import BaseDetector
from ..ocr import PlateOCR

class PlateDetector(BaseDetector):
    """Detector cho biến số xe"""
    def _load_model(self):
        with resources.path('lib.model', 'detect_PlateNumber.pt') as model_path:
            self.model = YOLO(str(model_path))
            self.ocr = PlateOCR()

    def _safe_crop(self, 
                   image: np.ndarray, 
                   x1: int, y1: int, x2: int, y2: int, 
                   pad: int = 8) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        h, w = image.shape[:2]
        x1_c = max(0, x1 - pad)
        y1_c = max(0, y1 - pad)
        x2_c = min(w, x2 + pad)
        y2_c = min(h, y2 + pad)
        if x2_c <= x1_c or y2_c <= y1_c:
            return image[0:0, 0:0], (x1, y1, x2, y2)
        return image[y1_c:y2_c, x1_c:x2_c], (x1, y1, x2, y2)


    def detect(self, image: np.ndarray) -> List[DetectionResult]: # type: ignore
        """
        Phát hiện biển số xe trong ảnh
        Args:
            image: Ảnh đầu vào (BGR format)
        Returns:
            List[DetectionResult]: Danh sách biển số phát hiện được
        """

        try:
            outs = self.model(image, conf=0.4, iou=0.5)
            res = outs[0] if isinstance(outs, (list, tuple)) else outs
            if not hasattr(res, "boxes") or len(res.boxes) == 0:
                return []

            boxes_xyxy = res.boxes.xyxy
            boxes_conf = res.boxes.conf
            if hasattr(boxes_xyxy, "cpu"): boxes_xyxy = boxes_xyxy.cpu().numpy()
            if hasattr(boxes_conf, "cpu"): boxes_conf = boxes_conf.cpu().numpy()

            candidates: list[tuple[float, PlateResult]] = []

            for (x1, y1, x2, y2), det_conf in zip(boxes_xyxy, boxes_conf):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                det_conf = float(det_conf)
                # Bỏ qua box không hợp lệ (sau khi ép int)
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped, _ = self._safe_crop(image, x1, y1, x2, y2, pad=8)

                failed_reason = None

                # -------- Check quality -----------
                box_w, box_h = x2 - x1, y2 - y1
                img_h, img_w = image.shape[:2]
                box_ratio = (box_w * box_h) / (img_w * img_h)

                # Too small (too far)
                if box_ratio < 0.003: # Càng tăng càng khắt khe
                    failed_reason = "[WARN] TOO FAR"

                # Too big (too close)
                elif box_ratio > 0.03: # Càng giảm càng khắt khe
                    failed_reason = "[WARN] TOO CLOSE"

                # Check blur
                elif cropped is not None and cropped.size > 0:
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if lap_var < 100:   # threshold tùy chỉnh
                        failed_reason = "[WARN] TOO BLUR"

                # Check skew (góc nghiêng)
                aspect_ratio = box_w / float(box_h + 1e-6)
                if aspect_ratio > 10 or aspect_ratio < 2.5:  # ISO code thường 2–8
                    failed_reason = "[WARN] TOO LEAN"
                
                # Nếu crop ra ảnh rỗng
                if cropped is None or cropped.size == 0:
                    r = PlateResult(
                        detection_type='plate',
                        bbox=[x1, y1, x2, y2],
                        confidence=det_conf,
                        text=None,
                        detection_confidence=det_conf,
                        ocr_confidence=0.0,
                        failed_reason=failed_reason,
                    )
                    candidates.append((det_conf, r))
                    continue

                try:
                    text, ocr_conf = self.ocr.extract_text(cropped)
                    if text is not None and isinstance(text, str):
                        text = text.strip()
                    ocr_conf = float(ocr_conf or 0.0)
                except Exception as e:
                    print(f"[ERROR] Plate OCR error: {e}")

                    text, ocr_conf = None, 0.0
                
                final_conf = ocr_conf if text else det_conf
                r = PlateResult(
                    detection_type='plate',
                    bbox=[x1, y1, x2, y2],
                    confidence=ocr_conf if text else det_conf,
                    text=text,
                    detection_confidence=det_conf,
                    ocr_confidence=ocr_conf if text else 0.0,
                    failed_reason=failed_reason,
                )
                candidates.append((final_conf, r))

            if not candidates:
                return []
            
            # Chọn best
            best_conf, best_result = max(candidates, key=lambda t: t[0])
            return [best_result]
    
        except Exception as e:
            print(f"[ERROR] Plate detection failed: {e}")
            return []


'''
* Các trường hợp gây lỗi detect, đọc dữ liệu sai:
+ Ảnh quá nhỏ
+ Ảnh quá xa
+ Ký tự bị cắt, tia sáng che chữ
+ Ảnh bị mờ
--> Phụ thuộc vào chất lượng ảnh
'''
