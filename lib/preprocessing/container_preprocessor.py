import cv2
import numpy as np
from typing import List
from .base_preprocessor import BasePreprocessor

class ContainerPreprocessor(BasePreprocessor):
    """Preprocessor nâng cao cho số container, sinh nhiều biến thể ảnh để tăng hiệu quả OCR.
       Không xoay ảnh vì camera cố định.
    """

    def __init__(self, deskew: bool = True):
        super().__init__('container')
        self.min_width = 400
        self.min_height = 150
        self.scale_factor_threshold = 2.0
        self.max_scale_factor = 3.0
        self.deskew = deskew

    def container_preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        processed_variants = []
        try:
            gray = self._validate_image(image)
            h, w = gray.shape

            # baseline gốc
            processed_variants.append(gray)

            # scale nếu ảnh quá nhỏ
            gray = self._scale_for_container(gray, h, w)

            # deskew nhẹ nếu bật
            if self.deskew:
                gray = self._deskew(gray)

            # 1. CLAHE
            clahe = self._apply_clahe(gray)
            processed_variants.append(clahe)

            # 2. Unsharp
            unsharp = self._apply_soft_unsharp(clahe)
            processed_variants.append(unsharp)

            # 3. Adaptive threshold
            adaptive = cv2.adaptiveThreshold(
                clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 25, 10
            )
            processed_variants.append(adaptive)

            # 4. Otsu threshold
            _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_variants.append(otsu)

            # 5. Gamma correction (sáng/tối khác nhau)
            for gamma in [0.7, 1.2]:
                gamma_corrected = np.array(
                    255 * ((clahe / 255.0) ** gamma),
                    dtype='uint8'
                )
                processed_variants.append(gamma_corrected)

            # 6. Top-hat & Black-hat
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            tophat = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel)
            processed_variants.extend([tophat, blackhat])

            # 7. Morphological gradient
            kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(clahe, cv2.MORPH_GRADIENT, kernel_grad)
            processed_variants.append(gradient)

            return processed_variants

        except Exception as e:
            print(f"[ERROR] Container preprocessing failed: {e}")
            return [gray] if 'gray' in locals() else []

    def _scale_for_container(self, gray: np.ndarray, h: int, w: int) -> np.ndarray:
        if h < self.min_height or w < self.min_width:
            scale = max(
                self.min_width / w,
                self.min_height / h,
                self.scale_factor_threshold
            )
            scale = min(scale, self.max_scale_factor)
            return cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LANCZOS4
            )
        return gray

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(img > 0))
        if len(coords) == 0:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _apply_soft_unsharp(self, img: np.ndarray) -> np.ndarray:
        gaussian = cv2.GaussianBlur(img, (3, 3), 0)
        unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        return np.clip(unsharp, 0, 255).astype(np.uint8)

    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError
