"""
Library - Thư viện phát hiện và nhận dạng đa đối tượng
Version: 1.0.0
"""

from lib.core.base_core import MultiDetect_Core
from .detectors import PlateDetector, ContainerDetector, FaceDetector
from .results import DetectionResult, PlateResult, ContainerResult, FaceResult
from .utils import ImageProcessor, BatchProcessor


__version__ = "1.0.0"
__author__ = "GitGud"

__all__ = [
    'MultiDetect_Core',
    'PlateDetector', 
    'ContainerDetector',
    'FaceDetector',
    'DetectionResult',
    'PlateResult',
    'ContainerResult', 
    'FaceResult',
    'ImageProcessor',
    'BatchProcessor',
]
