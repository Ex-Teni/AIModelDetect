# ==============CLEANER 11 CHARACTER================= #
import re
from typing import Optional, Tuple
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

        # Map ký tự dễ nhầm lẫn
        corrections = {
            '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'
        }
        digit_fallback = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1',
            'S': '5', 'Z': '2',
            'G': '6', 'B': '8'
        }

        # Nếu đủ dài → tách phần chữ và số
        if len(text) >= 10:
            letter_part = text[:4]
            number_part = text[4:]

            # Sửa chữ (phần đầu phải là 4 ký tự chữ)
            corrected_letters = ''.join(
                corrections.get(c, c) if c.isdigit() else c
                for c in letter_part
            )

            corrected_numbers = ''.join(
                digit_fallback.get(c, c) if c.isalpha() else c
                for c in number_part[:6]  # chỉ lấy 6 số
            )

            candidate = corrected_letters + corrected_numbers

            # Nếu match chuẩn → trả luôn
            if re.match(r'^[A-Z]{4}\d{6}$', candidate):
                check_digit = self._calculate_check_digit(candidate)
                return candidate + str(check_digit)

        # Nếu không chuẩn → fallback
        if len(text) >= 8:
            return text[:10]  # chỉ giữ 10 ký tự

        return None

    def _normalize_and_fix_orientation(self, text: str) -> Tuple[str, float]:
        """
        Chuẩn hóa text: remove space/hyphen, upper.
        Xử lý orientation nếu OCR đọc ngược block.
        Trả về (text_chuẩn, bonus).
        """
        t = (text or "").replace(" ", "").replace("-", "").upper()

        # Case chuẩn: 4 chữ + 6 số (10 ký tự, bỏ check digit)
        if re.match(r'^[A-Z]{4}\d{6}$', t):
            cd = self._calculate_check_digit(t)
            return t + str(cd), 0.30

        # Case OCR đọc ngược: 6 số + 4 chữ
        if re.match(r'^\d{6}[A-Z]{4}$', t):
            letters = t[-4:]
            numbers = t[:-4]
            fixed = letters + numbers
            if re.match(r'^[A-Z]{4}\d{6}$', fixed):
                cd = self._calculate_check_digit(fixed)
                return fixed + str(cd), 0.25
            return t, 0.10  # không fix được thì vẫn trả về

        # Case thiếu số: 4 chữ + 5 số
        if re.match(r'^[A-Z]{4}\d{5}$', t):
            return t, 0.15

        return t, 0.0


# # ==============CLEANER 10 CHARACTER================= #
# import re
# from typing import Optional, Tuple
# from .base_textcleaner import BaseTextCleaner

# class ContainerTextCleaner(BaseTextCleaner):
#     """Làm sạch text cho mã số container (ISO 6346 - bỏ check digit, chỉ lấy 10 ký tự)."""

#     def container_clean_text(self, text: str) -> Optional[str]:
#         if not text:
#             return None

#         # Chuẩn hóa ký tự
#         text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())

#         # Map ký tự dễ nhầm lẫn
#         corrections = {
#             '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'
#         }
#         digit_fallback = {
#             'O': '0', 'Q': '0', 'D': '0',
#             'I': '1', 'L': '1',
#             'S': '5', 'Z': '2',
#             'G': '6', 'B': '8'
#         }

#         # Nếu đủ dài → tách phần chữ và số
#         if len(text) >= 10:
#             letter_part = text[:4]
#             number_part = text[4:]

#             # Sửa chữ (phần đầu phải là 4 ký tự chữ)
#             corrected_letters = ''.join(
#                 corrections.get(c, c) if c.isdigit() else c
#                 for c in letter_part
#             )

#             corrected_numbers = ''.join(
#                 digit_fallback.get(c, c) if c.isalpha() else c
#                 for c in number_part[:6]  # chỉ lấy 6 số
#             )

#             candidate = corrected_letters + corrected_numbers

#             # Nếu match chuẩn → trả luôn
#             if re.match(r'^[A-Z]{4}\d{6}$', candidate):
#                 return candidate

#         # Nếu không chuẩn → fallback
#         if len(text) >= 8:
#             return text[:10]  # chỉ giữ 10 ký tự

#         return None

#     def _normalize_and_fix_orientation(self, text: str) -> Tuple[str, float]:
#         """
#         Chuẩn hóa text: remove space/hyphen, upper.
#         Xử lý orientation nếu OCR đọc ngược block.
#         Trả về (text_chuẩn, bonus).
#         """
#         t = (text or "").replace(" ", "").replace("-", "").upper()

#         # Case chuẩn: 4 chữ + 6 số (10 ký tự, bỏ check digit)
#         if re.match(r'^[A-Z]{4}\d{6}$', t):
#             return t, 0.30

#         # Case OCR đọc ngược: 6 số + 4 chữ
#         if re.match(r'^\d{6}[A-Z]{4}$', t):
#             letters = t[-4:]
#             numbers = t[:-4]
#             fixed = letters + numbers
#             if re.match(r'^[A-Z]{4}\d{6}$', fixed):
#                 return fixed, 0.25
#             return t, 0.10  # không fix được thì vẫn trả về

#         # Case thiếu số: 4 chữ + 5 số
#         if re.match(r'^[A-Z]{4}\d{5}$', t):
#             return t, 0.15

#         return t, 0.0