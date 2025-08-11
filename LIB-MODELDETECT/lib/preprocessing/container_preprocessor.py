import cv2
import numpy as np
from typing import List
from .base_preprocessor import BasePreprocessor

class ContainerPreprocessor(BasePreprocessor):
    """Preprocessor chuyên dụng cho số container"""
    
    def __init__(self):
        super().__init__('container')
        self.min_width = 400
        self.min_height = 150
        self.scale_factor_threshold = 2.5
    
    def container_preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Tiền xử lý ảnh cho OCR số container
        
        Vai trò:
        - Xử lý text container thường có font size lớn hơn, đơn dòng
        - Tối ưu cho container codes: format chữ-số, thường có nền kim loại
        - Xử lý các điều kiện ánh sáng khắc nghiệt (container ngoài trời)
        - Scale factor cao hơn vì container text cần độ phân giải cao
        """
        processed_variants = []
        
        try:
            gray = self._validate_image(image)
            h, w = gray.shape

            # Scale up mạnh hơn cho container text
            gray = self._scale_for_container(gray, h, w)
            h, w = gray.shape

            # 1. Baseline với CLAHE mạnh hơn
            enhanced = self._apply_strong_clahe(gray)
            processed_variants.append(enhanced)
            
            # 2. Bilateral filter + contrast enhancement
            enhanced_bilateral = self._enhance_with_bilateral(gray)
            processed_variants.append(enhanced_bilateral)
            
            # 3. Morphological operations cho container text
            cleaned = self._apply_container_morphology(enhanced)
            processed_variants.append(cleaned)
            
            # 4. Unsharp masking mạnh hơn
            unsharp = self._apply_strong_unsharp(enhanced)
            processed_variants.append(unsharp)
            
            # 5. Adaptive threshold variants cho container
            adaptive_variants = self._apply_container_adaptive_threshold(enhanced)
            processed_variants.extend(adaptive_variants)
            
            return processed_variants
            
        except Exception as e:
            print(f"[ERROR] Container preprocessing failed: {e}")
            return [gray] if 'gray' in locals() else []
    
    def _scale_for_container(self, gray: np.ndarray, h: int, w: int) -> np.ndarray:
        """Scale mạnh hơn cho container text"""
        if h < self.min_height or w < self.min_width:
            scale = max(
                self.min_width / w, 
                self.min_height / h, 
                self.scale_factor_threshold
            )
            return cv2.resize(
                gray, 
                (int(w * scale), int(h * scale)), 
                interpolation=cv2.INTER_LANCZOS4
            )
        return gray
    
    def _apply_strong_clahe(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE mạnh hơn cho container"""
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 2))
        return clahe.apply(gray)
    
    def _enhance_with_bilateral(self, gray: np.ndarray) -> np.ndarray:
        """Bilateral filter + histogram stretching"""
        bilateral = cv2.bilateralFilter(gray, 9, 80, 80)
        # Histogram stretching
        p2, p98 = np.percentile(bilateral, (2, 98)) # type: ignore
        bilateral_stretched = np.clip(
            (bilateral - p2) * 255.0 / (p98 - p2), 0, 255
        ).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 2))
        return clahe.apply(bilateral_stretched)
    
    def _apply_container_morphology(self, enhanced: np.ndarray) -> np.ndarray:
        """Morphological operations cho container text"""
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Opening để loại bỏ noise
        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_rect)
        # Closing để kết nối text bị gãy
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_rect)
    
    def _apply_strong_unsharp(self, enhanced: np.ndarray) -> np.ndarray:
        """Unsharp masking mạnh hơn cho container"""
        gaussian = cv2.GaussianBlur(enhanced, (3, 3), 0)
        unsharp = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
        return np.clip(unsharp, 0, 255).astype(np.uint8)
    
    def _apply_container_adaptive_threshold(self, enhanced: np.ndarray) -> List[np.ndarray]:
        """Adaptive threshold cho container với tham số mạnh hơn"""
        variants = []
        
        # Gaussian adaptive
        adaptive1 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        variants.append(adaptive1)
        
        # Mean adaptive
        adaptive2 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 19, 10
        )
        variants.append(adaptive2)
        
        return variants

    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError
