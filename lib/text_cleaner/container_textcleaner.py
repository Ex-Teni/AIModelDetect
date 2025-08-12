import re
from typing import Optional, Tuple
from .base_textcleaner import BaseTextCleaner

class ContainerTextCleaner(BaseTextCleaner):
    """Làm sạch text cho mã số container (ngang/dọc/vertical OCR)."""

    def container_clean_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        corrections = {
            '0': 'O', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'G': '6', 'B': '8',
            'Q': '0', 'D': '0', '@': '8', '?': '7', '%': '8'
        }
        if len(text) >= 10:
            letter_part = text[:4]
            number_part = text[4:]
            corrected_letters = ''
            for char in letter_part:
                if char.isdigit():
                    digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
                    corrected_letters += digit_to_letter.get(char, char)
                else:
                    corrected_letters += char
            corrected_numbers = ''
            for char in number_part[:6]:
                if char.isalpha():
                    corrected_numbers += corrections.get(char, '0')
                else:
                    corrected_numbers += char
            result = corrected_letters + corrected_numbers
            if re.match(r'^[A-Z]{4}\d{6,7}$', result):
                return result
        if len(text) >= 8:
            return text[:11]
        return None

    def _normalize_and_fix_orientation(self, text: str) -> Tuple[str, float]:
        """
        Chuẩn hóa text: remove space/hyphen, upper.
        Sửa orientation nếu match pattern đảo: 7 số + 4 chữ → đảo lại thành 4 chữ + 7 số.
        Trả về (text_chuẩn, bonus).
        """
        t = (text or "").replace(" ", "").replace("-", "").upper()
    
        # Pattern container ISO chuẩn: 4 chữ + 7 số
        if re.match(r'^[A-Z]{4}\d{7}$', t):
            return t, 0.30  # bonus cao vì match chuẩn
    
        # Pattern đảo: 7 số + 4 chữ (có thể do đọc ngược)
        if re.match(r'^\d{7}[A-Z]{4}$', t):
            rev = t[::-1]
            if re.match(r'^[A-Z]{4}\d{7}$', rev):
                return rev, 0.30  # đảo về chuẩn và áp bonus
            # Nếu đảo không ra chuẩn, vẫn trả t với bonus nhẹ để vote yếu
            return t, 0.10
    
        # Pattern “gần đúng”: 4 chữ + 6 số (thiếu 1 số)
        if re.match(r'^[A-Z]{4}\d{6}$', t):
            return t, 0.15
    
        # Không match gì đặc biệt
        return t, 0.0