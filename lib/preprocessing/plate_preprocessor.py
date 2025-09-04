import cv2
import numpy as np
from typing import List
from .base_preprocessor import BasePreprocessor

class PlatePreprocessor(BasePreprocessor):
    """Preprocessor chuyên dụng cho biển số xe Việt Nam (2 dòng)"""
    def __init__(self):
        super().__init__('plate')
        self.min_height = 80
        self.min_width = 200
        self.scale_factor_threshold = 2.0

    def plate_preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Tiền xử lý cho biển số xe 2 dòng
        
        Vai trò:
        - Xử lý ảnh biển số có kích thước nhỏ, độ tương phản thấp
        - Tối ưu cho đặc điểm biển số VN: text 2 dòng, nền trắng/xanh, chữ đen
        - Tạo multiple variants để tăng khả năng OCR thành công
        """

        processed_variants = []
        try:
            gray = self._validate_image(image)
            h, w = gray.shape

            # Scale up nếu ảnh quá nhỏ - quan trọng cho biển số nhỏ
            gray = self._scale_up(gray, h, w)
            h, w = gray.shape

            # 1.Baseline - CLAHE tối ưu cho biển số VN
            enhanced_clahe = self._apply_clahe(gray)
            processed_variants.append(enhanced_clahe)

            # 2.Xử lý ảnh độ tương phản thấp
            enhanced_low_contrast = self._enhance_low_contrast(gray)
            processed_variants.append(enhanced_low_contrast)

            # 3. Xử lý ảnh tối, thiếu sáng
            enhanced_low_light = self._enhance_low_light(gray)
            processed_variants.append(enhanced_low_light)

            # 4. Bilateral filter + sharpening cho text rõ nét
            enhanced_sharp = self._enhanced_sharp(gray)
            processed_variants.append(enhanced_sharp)

            # 5. Morphological operations để kết nối text bị đứt
            enhanced_morpho = self._enhanced_morpho(enhanced_clahe)
            processed_variants.append(enhanced_morpho)

            # 6.Adaptive thresholding variants
            adaptive_variants = self._adaptive_variants(enhanced_clahe)
            processed_variants.extend(adaptive_variants)

            return processed_variants[:7]
        
        except Exception as e:
            print(f"[ERROR] Plate preprocessing failed: {e}")
            return [gray] if 'gray' in locals() else []
        
    def _adaptive_variants(self, enhanced: np.ndarray) -> List[np.ndarray]:
        """Adaptive threshold với nhiều tham số"""
        variants = []
        
        # Gaussian adaptive
        adaptive1 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        variants.append(adaptive1)
        
        # Mean adaptive
        adaptive2 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 15, 3
        )
        variants.append(adaptive2)
        
        return variants

    def _enhanced_morpho(self, enhanced: np.ndarray) -> np.ndarray:
        """Morphological operations để làm sạch và kết nối text"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        # Opening để loại bỏ noise nhỏ
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        return cleaned

    def _enhanced_sharp(self, gray: np.ndarray) -> np.ndarray:
        """Tăng độ sắc nét cho ảnh"""
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        # Unsharp masking
        gaussian = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
        unsharp = cv2.addWeighted(bilateral, 1.5, gaussian, -0.5, 0)
        return unsharp

    def _enhance_low_light(self, gray: np.ndarray) -> np.ndarray:
        """Xử lý ảnh có độ sáng thấp"""
        gamma = 0.8  # Giảm gamma để tăng contrast
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(gray, gamma_table)
        return gamma_corrected

    def _enhance_low_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Xử lý ảnh có độ tương phản thấp"""
        equalized = cv2.equalizeHist(gray)
        clahe_lc = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        return clahe_lc.apply(equalized)

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE tối ưu cho biển số VN"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,2))
        return clahe.apply(gray)


    def _scale_up(self, gray: np.ndarray, h: int, w: int) -> np.ndarray:
        """Scale up image nếu ảnh nhỏ"""
        if h < self.min_height or w < self.min_width:
            scale_factor = max(
                self.min_height/h,
                self.min_width/w,
                self.scale_factor_threshold
            )
            new_w, new_h = int(w * scale_factor), int (h * scale_factor)
            return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return gray

    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError

