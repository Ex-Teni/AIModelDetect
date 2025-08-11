from typing import Optional

class BaseTextCleaner():
    """Base class cho các text cleaner."""
    def clean(self, text: str, *args, **kwargs) -> Optional[str]:
        raise NotImplementedError