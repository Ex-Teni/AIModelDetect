import numpy as np
import easyocr
from typing import Tuple, Optional
from paddleocr import PaddleOCR
import torch
from .base_ocr import BaseOCR
from ..preprocessing import ContainerPreprocessor
from ..text_cleaner import ContainerTextCleaner

class ContainerOCR(BaseOCR):
    """OCR chuyên biệt cho biển số xe"""
    
    def __init__(self):
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        self.easy_ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.preprocessor = ContainerPreprocessor()
        self.text_cleaner = ContainerTextCleaner()
    
    def extract_text(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Extract text từ biển số xe
        Args:
            image: Ảnh crop của biển số
        Returns:
            Tuple[text, confidence]: Text và độ tin cậy
        """
        try:
            # Preprocessing
            processed_variants = self.preprocessor.container_preprocess(image)
            
            all_results = []
            
            # Thử OCR với nhiều variant
            for variant_idx, processed_img in enumerate(processed_variants[:4]):
                
                # PaddleOCR
                try:
                    paddle_results = self.paddle_ocr.ocr(processed_img, det=True, rec=True, cls=True)
                    if paddle_results and paddle_results[0]:
                        text = self._extract_text_from_paddle_results(paddle_results[0])
                        if text:
                            cleaned = self.text_cleaner.container_clean_text(text)
                            if cleaned:
                                all_results.append((cleaned, 0.8, f"Paddle_v{variant_idx}"))
                except Exception as e:
                    print(f"[ERROR] PaddleOCR variant {variant_idx}: {e}")
                
                # EasyOCR  
                try:
                    easy_results = self.easy_ocr.readtext(processed_img, detail=1)
                    if easy_results:
                        text = ' '.join([item[1] for item in easy_results if item[2] > 0.1]) # type: ignore
                        if text:
                            cleaned = self.text_cleaner.container_clean_text(text)
                            if cleaned:
                                all_results.append((cleaned, 0.7, f"Easy_v{variant_idx}"))
                except Exception as e:
                    print(f"[ERROR] EasyOCR variant {variant_idx}: {e}")
            
            # Voting và chọn kết quả tốt nhất
            if all_results:
                # Score và sort
                scored_results = [(text, conf + self._calculate_pattern_bonus(text), method) 
                                for text, conf, method in all_results]
                scored_results.sort(key=lambda x: x[1], reverse=True)
                
                best_text, best_score, best_method = scored_results[0]
                print(f"[SUCCESS] Best container: {best_text} (score: {best_score:.3f}, method: {best_method})")
                return best_text, best_score
            
            return None, 0.0
            
        except Exception as e:
            print(f"[ERROR] Container OCR failed: {e}")
            return None, 0.0
    
    def _extract_text_from_paddle_results(self, paddle_results):
        """Utility để extract text từ PaddleOCR results"""
        texts = []
        for line in paddle_results:
            if line and len(line) >= 2:
                text, conf = line[1]
                if text and text.strip() and conf > 0.1:
                    texts.append(text.strip())
        return ' '.join(texts) if texts else None
    
    def _calculate_pattern_bonus(self, text: str) -> float:
        """Tính bonus điểm cho pattern đúng định dạng biển số VN"""
        import re
        clean_text = text.replace(' ', '').replace('-', '')
        if re.match(r'^\d{2}[A-Z]\d{4,6}$', clean_text):
            return 0.2
        elif re.match(r'^\d{2}[A-Z]{2}\d{4,6}$', clean_text):
            return 0.2
        return 0.0