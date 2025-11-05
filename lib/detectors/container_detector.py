from typing import List, Tuple
from importlib import resources
import numpy as np
import cv2
from ultralytics import YOLO

from ..results import DetectionResult, ContainerResult
from .base_detector import BaseDetector
from ..ocr import ContainerOCR

class ContainerDetector(BaseDetector):
    """Detector cho biến số xe"""

    # ------------------------------
    # Cấu hình chỉnh sửa
    # ------------------------------
    CONF_THRESHOLD   = 0.35      # Ngưỡng confidence YOLO
    IOU_THRESHOLD    = 0.5       # Ngưỡng IoU cho NMS
    PAD_SIZE         = 10        # Padding khi crop vùng container

    MIN_BOX_RATIO    = 0.003     # Tỉ lệ vùng bbox nhỏ nhất (so với ảnh)
    MAX_BOX_RATIO    = 0.01      # Tỉ lệ vùng bbox lớn nhất (so với ảnh)
    BLUR_THRESHOLD   = 100       # Ngưỡng Laplacian để kiểm tra độ mờ
    ASPECT_RATIO_MIN = 1.5       # Giới hạn tỉ lệ width/height thấp nhất
    ASPECT_RATIO_MAX = 8.0       # Giới hạn tỉ lệ width/height cao nhất


    def _load_model(self):
        with resources.path('lib.model', 'detect_ContainerCode.pt') as model_path:
            self.model = YOLO(str(model_path))
            self.ocr = ContainerOCR()

    def _safe_crop(self, 
                   image: np.ndarray, 
                   x1: int, y1: int, x2: int, y2: int, 
                   pad: int = 10) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
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
        Phát hiện mã container trong ảnh
        Args:
            image: Ảnh đầu vào (BGR format)
        Returns:
            List[DetectionResult]: Danh sách mã container phát hiện được
        """

        try:
            outs = self.model(image, conf=self.CONF_THRESHOLD, iou=self.IOU_THRESHOLD)
            res = outs[0] if isinstance(outs, (list, tuple)) else outs

            if not hasattr(res, "boxes") or res.boxes is None or len(res.boxes) == 0:
                return []

            boxes_xyxy = res.boxes.xyxy
            boxes_conf = res.boxes.conf
            if hasattr(boxes_xyxy, "cpu"): boxes_xyxy = boxes_xyxy.cpu().numpy()
            if hasattr(boxes_conf, "cpu"): boxes_conf = boxes_conf.cpu().numpy()

            candidates: List[Tuple[float, ContainerResult]] = []

            img_h, img_w = image.shape[:2]

            for (x1, y1, x2, y2), det_conf in zip(boxes_xyxy, boxes_conf):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                det_conf = float(det_conf)
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped, _ = self._safe_crop(image, x1, y1, x2, y2, pad=self.PAD_SIZE)
                failed_reason = None

                # ---------------------------
                # Kiểm tra chất lượng vùng phát hiện
                # ---------------------------
                box_w, box_h = x2 - x1, y2 - y1
                box_ratio = (box_w * box_h) / (img_w * img_h)

                if box_ratio < self.MIN_BOX_RATIO:
                    failed_reason = "[WARN] TOO FAR"
                elif box_ratio > self.MAX_BOX_RATIO:
                    failed_reason = "[WARN] TOO CLOSE"
                elif cropped is not None and cropped.size > 0:
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if lap_var < self.BLUR_THRESHOLD:
                        failed_reason = "[WARN] TOO BLUR"

                aspect_ratio = box_w / float(box_h + 1e-6)
                if aspect_ratio > self.ASPECT_RATIO_MAX or aspect_ratio < self.ASPECT_RATIO_MIN:
                    failed_reason = "[WARN] TOO LEAN"

                # ---------------------------
                # OCR xử lý vùng container
                # ---------------------------
                if cropped is None or cropped.size == 0:
                    r = ContainerResult(
                        detection_type='container',
                        bbox=[x1, y1, x2, y2],
                        confidence=det_conf,
                        text=None,
                        detection_confidence=det_conf,
                        ocr_confidence=0.0,
                        failed_reason=failed_reason,
                    )
                    candidates.append((det_conf, r))
                    continue

                text, ocr_conf = None, 0.0
                try:
                    text, ocr_conf = self.ocr.extract_text(cropped)
                    if text:
                        text = text.strip()
                        ocr_conf = float(ocr_conf or 0.0)
                except Exception as e:
                    print(f"[ERROR] Container OCR error: {e}")

                final_conf = ocr_conf if text else det_conf

                r = ContainerResult(
                    detection_type='container',
                    bbox=[x1, y1, x2, y2],
                    confidence=final_conf,
                    text=text,
                    detection_confidence=det_conf,
                    ocr_confidence=ocr_conf if text else 0.0,
                    failed_reason=failed_reason,
                )
                candidates.append((final_conf, r))

            if not candidates:
                return []

            best_conf, best_result = max(candidates, key=lambda t: t[0])
            return [best_result]

        except Exception as e:
            print(f"[ERROR] Container detection failed: {e}")
            return []