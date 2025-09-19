import cv2
import numpy as np
import easyocr
from typing import List, Tuple, Optional
from paddleocr import PaddleOCR
import torch
from .base_ocr import BaseOCR
from ..preprocessing import ContainerPreprocessor
from ..text_cleaner import ContainerTextCleaner

class ContainerOCR(BaseOCR):
    """OCR chuyên biệt cho mã container"""
    
    def __init__(self):
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        self.easy_ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.preprocessor = ContainerPreprocessor()
        self.text_cleaner = ContainerTextCleaner()
        self.min_token_conf = 0.1
    
    def extract_text(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Extract text từ số container
        Args:
            image: Ảnh crop của số container
        Returns:
            Tuple[text, confidence]: Text và độ tin cậy
        """
        try:
            processed_variants = self.preprocessor.container_preprocess(image)
            all_results = []

            for variant_idx, processed_img in enumerate(processed_variants[:4]):
                # PaddleOCR
                try:
                    paddle_results = self.paddle_ocr.ocr(processed_img, det=True, rec=True, cls=True)
                    if paddle_results and isinstance(paddle_results, list):
                        texts = self._extract_text_from_paddle_results(paddle_results)
                        if texts:
                            joined = ' '.join(texts)
                            cleaned = self.text_cleaner.container_clean_text(joined)
                            if cleaned:
                                t_norm, bonus = self.text_cleaner._normalize_and_fix_orientation(cleaned)
                                score = 0.8 + bonus
                                all_results.append((t_norm, float(score), f"Paddle_v{variant_idx}"))
                except Exception as e:
                    print(f"[ERROR] PaddleOCR variant {variant_idx}: {e}")

                # EasyOCR
                try:
                    easy_results = self.easy_ocr.readtext(processed_img, detail=1)
                    texts = []
                    for item in easy_results or []:
                        if not isinstance(item, (list, tuple)) or len(item) < 3:
                            continue
                        _, txt, conf = item
                        if isinstance(txt, str) and txt.strip() and isinstance(conf, (int, float)) and conf > self.min_token_conf:
                            texts.append(txt.strip())
                    
                    if texts:
                        joined = ' '.join(texts)
                        cleaned = self.text_cleaner.container_clean_text(joined)
                        if cleaned:
                            t_norm, bonus = self.text_cleaner._normalize_and_fix_orientation(cleaned)
                            score = 0.7 + bonus
                            all_results.append((t_norm, float(score), f"Paddle_v{variant_idx}"))
                    
                except Exception as e:
                    print(f"[ERROR] EasyOCR variant {variant_idx}: {e}")

            if all_results:
                # Score + pattern bonus
                scored_results = [
                    (text, float(conf) + float(self._calculate_pattern_bonus(text)), method)
                    for text, conf, method in all_results
                ]
                scored_results.sort(key=lambda x: x[1], reverse=True)
                best_text, best_score, best_method = scored_results[0]
                print(f"[SUCCESS] Best container: {best_text} (score: {best_score:.3f}, method: {best_method})")
                return best_text, float(best_score)

            return None, 0.0

        except Exception as e:
            print(f"[ERROR] Container OCR failed: {e}")
            return None, 0.0
    
    def _extract_text_from_paddle_results(self, paddle_results) -> List[str]:
        """Utility để extract text từ PaddleOCR results"""
        texts: List[str] = []
        try:
            for page in paddle_results:
                # page là list line cho ảnh hiện tại
                if not isinstance(page, (list, tuple)):
                    continue
                for line in page:
                    if (isinstance(line, (list, tuple)) and len(line) >= 2
                            and isinstance(line, (list, tuple)) and len(line) >= 2):
                        text, conf = line, line
                        if isinstance(text, str) and text.strip() and isinstance(conf, (int, float)) and conf > self.min_token_conf:
                            texts.append(text.strip())
        except Exception:
            # fallback: cấu trúc trả về có thể khác giữa các bản PaddleOCR → không chặn pipeline
            pass
        return texts
    
    def _calculate_pattern_bonus(self, text: str) -> float:
        """Tính bonus điểm cho pattern đúng định dạng số container"""
        import re
        clean_text = text.replace(' ', '').replace('-', '').upper()
        # Pattern: 4 chữ + 7 số
        if re.match(r'^[A-Z]{4}\d{7}$', clean_text):
            return 0.2
        return 0.0