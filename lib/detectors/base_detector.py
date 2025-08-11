from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union
import torch
import numpy as np

from ..results import DetectionResult

PathLike = Union[str, Path]

class BaseDetector(ABC):
    """Base class cho tất cả detectors"""
    
    def __init__(self, device: str):
        self.device = self._setup_device(device)
        self.model = None
        self._load_model()
    
    def _setup_device(self, device: str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @abstractmethod
    def _load_model(self):
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        pass