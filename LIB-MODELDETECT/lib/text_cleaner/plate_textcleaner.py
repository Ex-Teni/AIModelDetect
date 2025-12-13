import re
import csv
from typing import Dict, Optional, List
from importlib import resources
from difflib import SequenceMatcher

from .base_textcleaner import BaseTextCleaner


class PlateTextCleaner(BaseTextCleaner):
    """Làm sạch text chuyên biệt cho biển số xe Việt Nam."""

    def __init__(self):
        self.prefix_map = self._load_prefix_from_csv()

        # Các ký tự dễ nhầm OCR
        self.confusion_map = {
            '0': ['O', 'D', 'Q'],
            '1': ['I', 'L'],
            '2': ['Z'],
            '5': ['S'],
            '6': ['G'],
            '8': ['B'],
            'O': ['0', 'D'],
            'I': ['1', 'L'],
            'S': ['5'],
            'Z': ['2'],
            'G': ['6'],
            'B': ['8']
        }

    def _load_prefix_from_csv(self) -> Dict[str, Dict[str, List[str]]]:
        """Đọc bảng mã biển số xe Việt Nam."""
        prefix_map = {}
        try:
            with resources.open_text("lib.text_cleaner", "plate_prefix.csv", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 1 and row[0].strip().isdigit():
                        code = row[0].zfill(2)
                        prefix_map[code] = {
                            "single": [r.strip().upper() for r in row[1].split('|')] if len(row) > 1 and row[1].strip() else [],
                            "double": [r.strip().upper() for r in row[2].split('|')] if len(row) > 2 and row[2].strip() else []
                        }
        except Exception as e:
            print(f"[WARN] Cannot load plate prefix CSV: {e}")
        return prefix_map

    def _fix_with_prefix(self, text: str) -> str:
        """Tự động sửa phần đầu biển số dựa vào prefix_map."""
        if not text or len(text) < 4:
            return text

        text = text.upper().strip()
        first_two = re.sub(r'[^0-9]', '', text[:2])
        rest = text[2:]

        # Nếu 2 ký tự đầu không phải là số -> sửa
        if len(first_two) < 2:
            # thử thay thế ký tự sai bằng số gần đúng
            fixed_first = ''.join(
                c if c.isdigit() else
                next((k for k, v in self.confusion_map.items() if c in v and k.isdigit()), '0')
                for c in text[:2]
            )
            first_two = fixed_first

        # Lấy ký tự chữ (1 hoặc 2) theo template
        letter_candidates = []
        for code, region in self.prefix_map.items():
            if code == first_two:
                letter_candidates = region["single"] + region["double"]
                break

        letter_part = ''
        if len(rest) >= 1:
            sub = rest[:2]
            best_match, best_score = sub, -1
            for candidate in letter_candidates:
                score = SequenceMatcher(None, sub[:len(candidate)], candidate).ratio()
                if score > best_score:
                    best_match, best_score = candidate, score
            letter_part = best_match

        number_part = re.sub(r'[^0-9]', '', rest[len(letter_part):])
        number_part = ''.join(
            c if c.isdigit() else
            next((k for k, v in self.confusion_map.items() if c in v and k.isdigit()), '')
            for c in number_part
        )

        return first_two + letter_part + number_part

    def format_vietnam_plate(self, text: str) -> Optional[str]:
        """Định dạng lại biển số xe chuẩn Việt Nam."""
        if not text:
            return None

        text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        fixed_text = self._fix_with_prefix(text)

        # Kiểm tra định dạng hợp lệ: 2 số đầu, 1-2 chữ, 4-6 số
        if re.match(r'^\d{2}[A-Z]{1,2}\d{4,6}$', fixed_text):
            return fixed_text
        return fixed_text if 6 <= len(fixed_text) <= 10 else None

    def plate_clean_text(self, text: str, is_multiline: bool = False) -> Optional[str]:
        """Làm sạch chuỗi OCR biển số (1 dòng hoặc 2 dòng)."""
        if not text:
            return None

        text = re.sub(r'[^\w]', '', text.upper().strip())

        if is_multiline:
            lines = [line.strip() for line in re.split(r'[-\s]', text) if line.strip()]
            combined = ''.join(lines) if lines else text
        else:
            combined = text.replace('-', '').replace(' ', '')

        return self.format_vietnam_plate(combined)
