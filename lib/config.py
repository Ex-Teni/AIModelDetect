"""
Configuration settings cho MultiDetectLib
"""

import os
from typing import Dict, Any

# Model configurations
MODEL_CONFIG = {
    'plate': {
        'model_file': 'detect_PlateNumber.pt',
        'confidence_threshold': 0.4,
        'iou_threshold': 0.5,
    },
    'container': {
        'model_file': 'detect_ContainerCode.pt', 
        'confidence_threshold': 0.2,
        'iou_threshold': 0.5,
    },
    'face': {
        'detection_threshold': 0.1,
        'classification_threshold': 0.15,
    }
}

# OCR configurations
OCR_CONFIG = {
    'plate': {
        'timeout': 60,
        'confidence_threshold': 0.3,
        'preprocessing_variants': 7,
    },
    'container': {
        'timeout': 60,
        'confidence_threshold': 0.25,
        'preprocessing_variants': 6,
    }
}

# Processing configurations
PROCESSING_CONFIG = {
    'image_processing_timeout': 45,
    'max_workers': 4,
    'batch_size': 32,
}

# Device configurations
DEVICE_CONFIG = {
    'auto_select': True,
    'prefer_gpu': True,
    'fallback_cpu': True,
}

def get_config() -> Dict[str, Any]:
    """Lấy toàn bộ configuration"""
    return {
        'models': MODEL_CONFIG,
        'ocr': OCR_CONFIG,
        'processing': PROCESSING_CONFIG,
        'device': DEVICE_CONFIG,
    }
