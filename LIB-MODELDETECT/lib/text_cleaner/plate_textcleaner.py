import re
from typing import Optional
from .base_textcleaner import BaseTextCleaner

class PlateTextCleaner(BaseTextCleaner):
    """Làm sạch text chuyên biệt cho biển số xe Việt Nam."""

    def format_vietnam_plate(self, text: str) -> Optional[str]:
        if not text:
            return None
        
        text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())

        to_digit = {'O': '0', 'D': '0', 'Q': '0', 'B': '8', 'S': '5', 'Z': '2', 'G': '6', 'I': '1', 'L': '1'}
        to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
        
        chars = list(text)
        
        for i in range(min(2, len(chars))):
            if chars[i] in to_digit:
                chars[i] = to_digit[chars[i]]
        if len(chars) >= 3 and chars[2] in to_letter:
            chars[2] = to_letter[chars[2]]
        if len(chars) >= 4 and not chars[3].isdigit():
            if chars[3] in to_letter:
                chars[3] = to_letter[chars[3]]
            elif chars[3] in to_digit:
                chars[3] = to_digit[chars[3]]
        start_idx = 4 if len(chars) >= 4 and chars[3].isalpha() else 3
        for i in range(start_idx, len(chars)):
            if chars[i] in to_digit:
                chars[i] = to_digit[chars[i]]
        result = ''.join(chars)
        if re.match(r'^\d{2}[A-Z]\d{4,6}$', result) or re.match(r'^\d{2}[A-Z]{2}\d{4,6}$', result):
            return result
        return result if 6 <= len(result) <= 10 else None

    def plate_clean_text(self, text: str, is_multiline: bool = False) -> Optional[str]:
        if not text:
            return None
        text = re.sub(r'[^\w]', '', text.upper().strip())
        ocr_corrections = {
            'O': '0', 'I': '1', 'L': '1', '|': '1', 'S': '5', 'Z': '2', 'G': '6',
            'B': '8', 'Q': '0', 'D': '0', '@': '8', '?': '7', '%': '8', '&': '8',
            ' ': '', '\t': '', '\n': ' ' if is_multiline else ''
        }
        for wrong, correct in ocr_corrections.items():
            text = text.replace(wrong, correct)
        if is_multiline:
            lines = [line.strip() for line in text.split('-') if line.strip()]
            if len(lines) >= 2:
                combined = ''.join(lines)
            elif len(lines) == 1:
                combined = lines[0]
            else:
                combined = text.replace('-', '')
        else:
            combined = text.replace('-', '')
        return self.format_vietnam_plate(combined)
