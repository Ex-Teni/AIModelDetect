import csv
import cv2
import numpy as np
import easyocr
from typing import List, Tuple, Optional
from paddleocr import PaddleOCR
import torch
from rapidfuzz import process, fuzz
from importlib import resources

from .base_ocr import BaseOCR
from ..preprocessing import ContainerPreprocessor
from ..text_cleaner import ContainerTextCleaner

class ContainerOCR(BaseOCR):
    """OCR chuyên biệt cho mã container + prefix ISO 4 ký tự"""
    
    def __init__(self, prefix_csv: Optional[str] = None):
        "Khởi tạo các biến dữ liệu, các module trong lib, file csv chứa các mã ISO 4 ký tự"
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        self.easy_ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.preprocessor = ContainerPreprocessor()
        self.text_cleaner = ContainerTextCleaner()
        self.min_token_conf = 0.1
        self.prefix_list = self._load_prefix_from_csv(prefix_csv)

    def _load_prefix_from_csv(self, csv_path: str) -> List[str]:
        "Load và đọc các mã ISO trong file csv"
        prefixes = []
        try:
            with resources.open_text("lib.ocr", "container_prefix.csv", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        val = cell.strip().upper()
                        if val and len(val) == 4:
                            prefixes.append(val)
        except Exception as e:
            print(f"[WARN] Could not load prefix CSV: {e}")
        return list(set(prefixes))
    
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
                            fixed = self._fix_with_prefix(joined)
                            if fixed:
                                bonus = self._calculate_pattern_bonus(fixed)
                                score = 0.8 + bonus
                                all_results.append((fixed, float(score), f"Paddle_v{variant_idx}"))
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
                        fixed = self._fix_with_prefix(joined)
                        if fixed:
                            bonus = self._calculate_pattern_bonus(fixed)
                            score = 0.7 + bonus
                            all_results.append((fixed, float(score), f"Easy_v{variant_idx}"))
                    
                except Exception as e:
                    print(f"[ERROR] EasyOCR variant {variant_idx}: {e}")

            if all_results:
                # Chọn kết quả tốt nhất
                best_text, best_score, best_method = max(all_results, key=lambda x: x[1])
                # Sau khi đã fix prefix thì mới đưa qua text_cleaner
                cleaned = self.text_cleaner.container_clean_text(best_text)
                print(f"[SUCCESS] Best container: {cleaned} (score: {best_score:.3f}, method: {best_method})")
                return cleaned, float(best_score)

            return None, 0.0

        except Exception as e:
            print(f"[ERROR] Container OCR failed: {e}")
            return None, 0.0
    
    def _extract_text_from_paddle_results(self, paddle_results) -> List[str]:
        """Utility để extract text từ PaddleOCR results"""
        texts: List[str] = []
        try:
            for page in paddle_results:
                if not isinstance(page, (list, tuple)):
                    continue
                for line in page:
                    # line = [bbox, (text, conf)]
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        if isinstance(line[1], (list, tuple)) and len(line[1]) == 2:
                            text, conf = line[1]
                            if (isinstance(text, str) and text.strip() and
                                isinstance(conf, (int, float)) and conf > self.min_token_conf):
                                texts.append(text.strip())
        except Exception:
            pass
        return texts

    def _fix_with_prefix(self, text: str) -> str:
        """
        Sửa prefix của mã container dựa trên danh sách prefix ISO
        """
        if len(text) < 4:
            return text

        raw_prefix, suffix = text[:4].upper(), text[4:]

        if raw_prefix in self.prefix_list:
            return raw_prefix + suffix

        prefix, suffix = text[:4].upper(), text[4:]
        best_match, best_score = None, -1

        for candidate in self.prefix_list:
            if len(candidate) != 4:
                continue
            
            # Đếm số ký tự trùng
            match_count = sum(p == c for p, c in zip(prefix, candidate))

            # Tính thêm điểm nếu khác biệt nằm trong confusion_map
            bonus = 0
            for p, c in zip(prefix, candidate):
                if p != c and p in self.confusion_map and c in self.confusion_map[p]:
                    bonus += 0.5  # cộng điểm "hợp lý"

            score = match_count + bonus

            # Chọn ứng viên có score cao nhất
            if score > best_score:
                best_match, best_score = candidate, score

        if best_match:
            return best_match + suffix
        return text
    
    def _calculate_pattern_bonus(self, text: str) -> float:
        """Tính bonus điểm cho pattern đúng định dạng số container"""
        import re
        clean_text = text.replace(' ', '').replace('-', '').upper()
        # Pattern: 4 chữ + 7 số
        if re.match(r'^[A-Z]{4}\d{7}$', clean_text):
            return 0.2
        return 0.0
