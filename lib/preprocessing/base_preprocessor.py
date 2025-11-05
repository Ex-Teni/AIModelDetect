from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List

class BasePreprocessor (ABC):
    """Base class cho tất cả preprocess"""
    def __init__ (self, target_type: str):
        """
        Args:
            target_type: Loại đối tượng cần xử lý ('plate' hoặc 'container')
        """
        self.target_type = target_type
    
    @abstractmethod
    def preprocess (self, image: np.ndarray) -> List[np.ndarray]:
        """Xử lý ảnh và trả về danh sách các variant đã xử lý"""
        pass

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validate và convert image sang grayscale"""
        if image is None or image.size ==0:
            raise ValueError("Input image is empty")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image
    

