import re
from typing import Optional
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
