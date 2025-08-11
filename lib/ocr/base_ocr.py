from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class BaseOCR(ABC):
    "Base class cho OCR"
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Extract text từ ảnh"""
        pass
    