# ==============CLEANER 11 CHARACTER================= #
import re
import csv
from typing import  List, Optional, Tuple
from importlib import resources

from .base_textcleaner import BaseTextCleaner

class ContainerTextCleaner(BaseTextCleaner):
    """Làm sạch text cho mã số container (ISO 6346 - bỏ check digit, chỉ lấy 10 ký tự)."""

    CHAR_VALUES = {
        'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15,
        'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
        'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26,
        'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31,
        'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
    }

    def __init__(self, prefix_csv: Optional[str] = None):
        self.prefix_list = self._load_prefix_from_csv(prefix_csv)

    def _load_prefix_from_csv(self, csv_path: Optional[str] = None) -> List[str]:
        """Đọc prefix ISO container từ CSV."""
        prefixes = []
        try:
            path_to_use = csv_path or "container_prefix.csv"
            with resources.open_text("lib.text_cleaner", path_to_use, encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        val = cell.strip().upper()
                        if val and len(val) == 4:
                            prefixes.append(val)
        except Exception as e:
            print(f"[WARN] Cannot load prefix CSV: {e}")
        return list(set(prefixes))
    
    def _fix_with_prefix(self, text: str) -> str:
        if not text or len(text) < 4:
            return text
        
        prefix, suffix = text[:4].upper(), text[4:]
        if prefix in self.prefix_list:
            return prefix + suffix
        
        best_match, best_score = None, -1
        for candidate in self.prefix_list:
            if len(candidate) != 4:
                continue

            match_count = sum(p == c for p, c in zip(prefix, candidate))
            bonus = sum(
                0.5 for p, c in zip(prefix, candidate)
                if p != c and p in self.confusion_map and c in self.confusion_map[p]
            )

            score = match_count + bonus
            if score > best_score:
                best_match, best_score = candidate, score

        return (best_match + suffix) if best_match else text

    def _calculate_check_digit(self, code10: str) -> int:
        total = 0
        for i,c in enumerate(code10):
            if c.isdigit():
                value = int(c)
            else:
                value = self.CHAR_VALUES.get(c, 0)
            total += value * (2 **i)
        check_digit = total % 11
        return 0 if check_digit == 10 else check_digit


    def container_clean_text(self, text: str) -> Optional[str]:
        if not text:
            return None

        # Chuẩn hóa ký tự
        text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        # Nếu OCR đọc ngược (6 số trước, 4 chữ sau) → đảo lại
        if re.match(r'^\d{6}[A-Z]{4}$', text):
            text = text[-4:] + text[:-4]
        text = self._fix_with_prefix(text)

        # Map ký tự dễ nhầm lẫn
        to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
        to_digit = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'G': '6', 'B': '8'}

        # Nếu đủ dài → tách phần chữ và số
        if len(text) >= 10:
            letter_part = ''.join(
                to_letter.get(c, c) if c.isdigit() else c
                for c in text[:4]
            )
            number_part = ''.join(
                to_digit.get(c, c) if c.isalpha() else c
                for c in text[4:]
            )

            # Giới hạn phần số tối đa 6 ký tự
            number_part = re.sub(r'\D', '', number_part)[:6]

            if len(letter_part) == 4 and len(number_part) == 6:
                base = letter_part + number_part
                check_digit = self._calculate_check_digit(base)
                return base + str(check_digit)

        # Fallback: nếu không hợp lệ nhưng vẫn có 8–10 ký tự
        if len(text) >= 8:
            clean_fallback = text[:10]
            if re.match(r'^[A-Z]{3,4}\d{5,6}$', clean_fallback):
                cd = self._calculate_check_digit(clean_fallback[:10])
                return clean_fallback[:10] + str(cd)
            return clean_fallback

        return None