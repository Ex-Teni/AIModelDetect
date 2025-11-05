import re
from typing import Optional

class BaseTextCleaner():
    """Base class cho các text cleaner."""

    confusion_map = {
        '0': 'O', 'O': '0',
        '1': 'I', 'I': '1', 'L': '1',
        '2': 'Z', 'Z': '2',
        '5': 'S', 'S': '5',
        '6': 'G', 'G': '6',
        '8': 'B', 'B': '8',
        '9': 'G'
    }

    def normalize(self, text: str) -> str:
        """Chuẩn hóa text: bỏ ký tự lạ, viết hoa."""
        if not text:
            return ""
        text = text.strip().upper()
        return re.sub(r'[^A-Z0-9]', '', text)

    def map_confusion(self, text: str) -> str:
        """Sửa ký tự dễ nhầm lẫn."""
        return ''.join(self.confusion_map.get(c, c) for c in text)

    def clean(self, text: str, *args, **kwargs) -> Optional[str]:
        raise NotImplementedError