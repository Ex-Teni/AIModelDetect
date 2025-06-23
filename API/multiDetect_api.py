import asyncio
import subprocess
import time
import cv2
import torch
import json
import base64
import numpy as np
import joblib
import re
import easyocr
import threading
import pytesseract
import concurrent.futures
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
from functools import wraps
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from starlette.websockets import WebSocketDisconnect
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass

# ===== GLOBAL VARIABLES =====
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class GlobalState:
    def __init__(self):
        self.latest_frame = None
        self.latest_metadata = {"plate": "None", "container": "None", "face": "None", "seal": "None"}
        self.frame_lock = threading.Lock()
        self.models_loaded = False

# Detection thresholds
FACE_CONFIDENCE_THRESHOLD = 0.95  # Face detection threshold
OCR_CONFIDENCE_THRESHOLD = 0.3    # OCR confidence threshold
SEAL_OCR_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for seal detection

# WebSocket timeout settings
WEBSOCKET_TIMEOUT = 30  # seconds
PING_INTERVAL = 10      # seconds
PING_TIMEOUT = 5        # seconds

# Processing timeout settings
IMAGE_PROCESSING_TIMEOUT = 25  # seconds (less than WebSocket timeout)
OCR_TIMEOUT = 5  # seconds per OCR operation

# ===== CORS MIDDLEWARE =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODEL INITIALIZATION =====
class ModelManager:
    def __init__(self):
        self.yolo_models = {}
        self.face_models = {}
        self.ocr_models = None 
        self.face_classifier = None 
        self.label_encoder = None 
        self.face_transform = None
    
    def load_models(self):
        """Gọi model từ folde modelAI"""
        try:
            print(f"[INFO] Loading all model...")
            self.yolo_models['plate'] = YOLO("modelAI/detect_PlateNumber.pt")
            self.yolo_models['container'] = YOLO("modelAI/detect_ContainerCode.pt")
            self.yolo_models['character'] = YOLO("modelAI/detect_Character.pt")
            self.face_models['mtcnn'] = MTCNN(keep_all=True, device=device)
            self.face_models['facenet'] = InceptionResnetV1(pretrained='vggface2').eval().to(device)

            try: 
                self.face_classifier = joblib.load('modelAI/face_classifier.joblib')
                self.label_encoder = joblib.load('modelAI/label_encoder.joblib')
                print("[INFO] Face classifier loaded successfully")
            except Exception as e:
                print(f"[WARNING] Face classifier not found: {e}")
                self.face_classifier = None
                self.label_encoder = None

            self.face_transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            self.ocr_models = OCRModels()
            print(f"[INFO] All models load successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            return False
        
global_state = GlobalState()
model_manager = ModelManager()

# Load models khi khởi động
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[STARTUP] Loading models...")
    success = model_manager.load_models()
    if success:
        global_state.models_loaded = True
        print("[STARTUP] All models loaded successfully")
    else:
        print("[STARTUP] Failed to load models")
    yield
    # Shutdown
    print("[SHUTDOWN] Server shutting down")
app = FastAPI(lifespan=lifespan)

# OCR init
class OCRModels:
    def __init__(self):

        # PaddleOCR init
        try:
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                # Removed show_log parameter as it's not supported
            )
            print("[INFO] PaddleOCR initialized successfully")
        except Exception as e:
            print(f"[ERROR] PaddleOCR initialization failed: {e}")
            self.paddle_ocr = None

        self.easy_ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        # TrOCR fallback
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            print("[INFO] TrOCR models loaded successfully")
        except Exception as e:
            self.trocr_processor = None
            self.trocr_model = None
            print(f"[WARNING] TrOCR models failed to load: {e}")

        # Tesseract config
        self.plate_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        self.container_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

@dataclass
class PlateResult:
    text: str
    confidence: float
    method: str
    is_multiline: bool


# ===== UTILITY FUNCTIONS =====
def decode_base64_to_image(b64: str) -> Optional[np.ndarray]:
    """Decode base64 string to OpenCV image"""
    try:
        img_data = base64.b64decode(b64)
        arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] Base64 decode failed: {e}")
        return None

def encode_image_to_base64(img: np.ndarray) -> str:
    """Encode OpenCV image to base64 string"""
    _, buff = cv2.imencode(".jpg", img)
    return base64.b64encode(buff).decode("utf-8") # type: ignore

def timeout_with_executor(timeout_seconds: float):
    """Windows-compatible timeout decorator using ThreadPoolExecutor"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(func, *args, **kwargs)
                    result = future.result(timeout=timeout_seconds)
                    return result
                except concurrent.futures.TimeoutError:
                    print(f"[TIMEOUT] {func.__name__} timed out after {timeout_seconds}s")
                    return None, 0.0 if func.__name__.startswith('extract_text') else None
                except Exception as e:
                    print(f"[ERROR] {func.__name__} failed: {e}")
                    return None, 0.0 if func.__name__.startswith('extract_text') else None
        return wrapper
    return decorator

def boxes_overlap(box1, box2, threshold=0.3):
    """Check if two bounding boxes overlap above threshold"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    if min(box1_area, box2_area) <= 0:
        return False
    
    overlap_ratio = intersection_area / min(box1_area, box2_area)
    return overlap_ratio > threshold

def remove_duplicate_detections(detections):
    """Remove duplicate and overlapping detections"""
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    filtered = []
    for det in detections:
        is_duplicate = False
        for existing in filtered:
            # Check if boxes overlap significantly
            if boxes_overlap(det['box'], existing['box'], threshold=0.5):
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(det)
    
    return filtered

# ===== TEXT PROCESSING FUNCTIONS =====
def preprocess_image_for_ocr(image):
    """Preprocess image to improve OCR accuracy"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize if too small
    height, width = gray.shape
    if height < 64 or width < 200:
        scale = max(64 / height, 200 / width, 2.0)
        new_height = int(height * scale)
        new_width = int(width * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply multiple preprocessing methods and choose the best
    processed_images = []
    
    # Method 1: Denoising + CLAHE
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    processed_images.append(enhanced)
    
    # Method 2: Gaussian blur + OTSU threshold
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(thresh)

    # Method 3: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    processed_images.append(morph)
    
    # Use the first processed image (can be enhanced with quality assessment)
    best_image = processed_images[0]
    
    # Convert to RGB for TrOCR
    rgb_image = cv2.cvtColor(best_image, cv2.COLOR_GRAY2RGB)
    return rgb_image

def detect_text_orientation(image: np.ndarray) -> str:
    """Phát hiện hướng của text (horizontal/vertical)"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=20)
        
        if lines is None:
            return "horizontal"
        
        horizontal_count = 0
        vertical_count = 0

        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi

            if angle < 10 or angle > 170:
                horizontal_count += 1
            elif 80 < angle < 100:
                vertical_count += 1
        
        return "vertical" if vertical_count > horizontal_count else "horizontal"
    except Exception as e:
        print(f"[FAILED] Text orientation detection failed: {e}")
        return "horizontal"

def preprocess_for_multiline_text(image: np.ndarray) -> List[np.ndarray]:
    """Tiền xử lý chuyên biệt cho biển số xe 2 dòng"""
    processed_variants = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # Resize nếu quá nhỏ
        if h < 80 or w < 200:
            scale = max(80/h, 200/w, 1.5)
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            h, w = gray.shape
        
        # Variant 1: CLAHE + Gaussian blur
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        gaussian = cv2.GaussianBlur(enhanced, (3,3), 0)
        processed_variants.append(gaussian)
        
        # Variant 2: Denoising + Sharpening
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
        processed_variants.append(sharpened)
        
        # Variant 3: Morphological operations for text connection
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        morph_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)
        processed_variants.append(morph_close)
        
        # Variant 4: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed_variants.append(adaptive)
        
        # Variant 5: OTSU threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_variants.append(otsu)
        
        return processed_variants
        
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return [gray] if 'gray' in locals() else []

def preprocess_for_vertical_text(image):
    """Tiền xử lý hình ảnh cho văn bản đọc"""
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize if too small
    h, w = gray.shape
    if h < 100:
        scale = 100 / h
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # Enhance contrast specifically for vertical text
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 8))
    enhanced = clahe.apply(gray)
    
    # Vertical morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    return morph

def detect_plate_layout(image: np.ndarray) -> Dict[str, any]: # type: ignore
    """Phát hiện layout của biển số xe (1 dòng hay 2 dòng)"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Tìm contours để phát hiện text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations để kết nối các ký tự
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        
        # Tìm contours của các dòng text
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc contours theo kích thước
        text_regions = []
        h, w = gray.shape
        min_area = (w * h) * 0.02  # Tối thiểu 2% diện tích
        
        for contour in contours:
            x, y, w_c, h_c = cv2.boundingRect(contour)
            area = w_c * h_c
            
            if area > min_area and w_c > 30 and h_c > 10:
                text_regions.append({
                    'bbox': (x, y, x + w_c, y + h_c),
                    'area': area,
                    'center_y': y + h_c // 2
                })
        
        # Sắp xếp theo vị trí Y
        text_regions.sort(key=lambda x: x['center_y'])
        
        # Phân tích layout
        is_multiline = len(text_regions) >= 2
        aspect_ratio = w / h if h > 0 else 1
        
        return {
            'is_multiline': is_multiline,
            'text_regions': text_regions,
            'aspect_ratio': aspect_ratio,
            'num_regions': len(text_regions)
        }
        
    except Exception as e:
        print(f"[ERROR] Layout detection failed: {e}")
        return {'is_multiline': False, 'text_regions': [], 'aspect_ratio': 1, 'num_regions': 0}
    

def extract_text_regions_separately(image: np.ndarray, layout_info: Dict) -> List[Tuple[np.ndarray, str]]:
    """Tách các vùng text riêng biệt để OCR"""
    text_crops = []
    
    try:
        if not layout_info['is_multiline'] or not layout_info['text_regions']:
            return [(image, "single_line")]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        for i, region in enumerate(layout_info['text_regions']):
            x1, y1, x2, y2 = region['bbox']
            
            # Thêm padding nhỏ
            padding = 3
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(gray.shape[1], x2 + padding)
            y2 = min(gray.shape[0], y2 + padding)
            
            if x2 > x1 and y2 > y1:
                crop = gray[y1:y2, x1:x2]
                text_crops.append((crop, f"line_{i+1}"))
        
        return text_crops
        
    except Exception as e:
        print(f"[ERROR] Text region extraction failed: {e}")
        return [(image, "single_line")]
    

def clean_plate_text(text: str, is_multiline: bool = False) -> Optional[str]:
    """Cải tiến text cleaning cho biển số xe"""
    if not text:
        return None
    
    # Loại bỏ ký tự đặc biệt và chuẩn hóa
    text = re.sub(r'[^\w\s\-\.]', '', text.upper().strip())
    
    # Sửa lỗi OCR
    ocr_corrections = {
        'O': '0', 'I': '1', 'l': '1', '|': '1',
        'S': '5', 'Z': '2', 'G': '6', 'B': '8',
        'Q': '0', 'D': '0'
    }
    
    for wrong, correct in ocr_corrections.items():
        text = text.replace(wrong, correct)
    
    if is_multiline:
        # Xử lý biển số 2 dòng
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) >= 2:
            # Ghép các dòng với dấu gạch ngang
            line1 = re.sub(r'[^A-Z0-9]', '', lines[0])
            line2 = re.sub(r'[^A-Z0-9]', '', lines[1])
            
            if line1 and line2:
                combined = f"{line1}-{line2}"
                if len(combined) >= 4:
                    return combined
        
        # Fallback: ghép tất cả thành 1 chuỗi
        combined = ''.join(re.sub(r'[^A-Z0-9]', '', line) for line in lines)
        if len(combined) >= 4:
            return combined
    
    else:
        # Xử lý biển số 1 dòng
        cleaned = re.sub(r'[^A-Z0-9\-]', '', text)
        if len(cleaned) >= 2:
            return cleaned
    
    return None

def clean_container_text(text: str) -> Optional[str]:
    """Xử lý text container code dọc"""
    # Container format: 4 letters + 7 digits (e.g., ABCD1234567)
    text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
    
    # Common OCR corrections
    text = text.replace('O', '0').replace('I', '1').replace('S', '5')
    
    # Validate container format (at least 4 characters)
    if len(text) >= 4:
        return text
    
    return None
def clean_text(text: str, text_type: str) -> Optional[str]:
    """Clean and validate OCR text based on type"""
    if not text:
        return None
    
    # Convert to uppercase and remove leading/trailing spaces
    text = text.strip().upper()

    if text_type == "plate":
        # License plate: keep only letters, numbers, and hyphens
        text = re.sub(r'[^A-Z0-9\-]', '', text)
        if len(text) < 2:
            return None
        
    elif text_type == "container":
        # Container code: keep only letters and numbers
        text = re.sub(r'[^A-Z0-9]', '', text)
        # Common OCR corrections
        text = text.replace('O', '0').replace('I', '1')
        if len(text) < 4:
            return None
        
    elif text_type == "seal":
        # Seal: more flexible, allow word characters, hyphens, and dots
        text = re.sub(r'[^\w\-\.]', '', text)
        if len(text) < 2:
            return None
    
    return text if text else None

# ===== OCR FUNCTIONS =====
@timeout_with_executor(OCR_TIMEOUT)
def extract_text_with_easyocr_fast(image_crop: np.ndarray, text_type: str = "plate") -> Tuple[Optional[str], float]:
    """Fast EasyOCR extraction as fallback"""
    try:
        if model_manager.ocr_models is None or model_manager.ocr_models.easy_ocr is None:
            return None, 0.0
            
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
            
        results = model_manager.ocr_models.easy_ocr.readtext(gray, detail=1, width_ths=0.7, height_ths=0.7)
        if not results:
            return None, 0.0
            
        best_result = max(results, key=lambda x: x[2]) # type: ignore
        text = best_result[1] # type: ignore
        confidence = best_result[2] # type: ignore
        cleaned_text = clean_text(text, text_type)
        return cleaned_text, confidence if cleaned_text else 0.0 # type: ignore
        
    except Exception as e:
        print(f"[FAILED] Fast EasyOCR failed: {e}")
        return None, 0.0

@timeout_with_executor(OCR_TIMEOUT)
def extract_text_plate(image_crop: np.ndarray, ocr_models) -> Tuple[Optional[str], float]:
    """Cải tiến OCR cho biển số xe 2 dòng"""
    all_results = []
    
    try:
        # 1. Phát hiện layout
        layout_info = detect_plate_layout(image_crop)
        is_multiline = layout_info['is_multiline']
        
        print(f"[INFO] Detected layout: {'Multiline' if is_multiline else 'Single line'}")
        
        # 2. Tiền xử lý chuyên biệt
        processed_variants = preprocess_for_multiline_text(image_crop)
        
        # 3. Tách vùng text nếu là multiline
        if is_multiline:
            text_regions = extract_text_regions_separately(image_crop, layout_info)
        else:
            text_regions = [(image_crop, "single_line")]
        
        # 4. OCR với nhiều phương pháp
        for variant_idx, processed_img in enumerate(processed_variants):
            
            # Method 1: PaddleOCR
            if hasattr(ocr_models, 'paddle_ocr') and ocr_models.paddle_ocr is not None:
                try:
                    paddle_results = ocr_models.paddle_ocr.ocr(processed_img, cls=True)
                    if paddle_results and paddle_results[0]:
                        if is_multiline:
                            # Xử lý nhiều dòng
                            lines = []
                            for line in paddle_results[0]:
                                lines.append(line[1][0])
                            text = '\n'.join(lines)
                        else:
                            text = paddle_results[0][0][1][0]
                        
                        conf = paddle_results[0][0][1][1]
                        cleaned = clean_plate_text(text, is_multiline)
                        
                        if cleaned and conf > 0.7:
                            all_results.append(PlateResult(
                                text=cleaned,
                                confidence=conf,
                                method=f"PaddleOCR_v{variant_idx}",
                                is_multiline=is_multiline
                            ))
                            
                except Exception as e:
                    print(f"[ERROR] PaddleOCR variant {variant_idx} failed: {e}")
            
            # Method 2: EasyOCR với cấu hình multiline
            if hasattr(ocr_models, 'easy_ocr') and ocr_models.easy_ocr is not None:
                try:
                    easy_results = ocr_models.easy_ocr.readtext(
                        processed_img,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                        width_ths=0.7,
                        height_ths=0.4 if is_multiline else 0.7,
                        paragraph=is_multiline
                    )
                    
                    if easy_results:
                        if is_multiline and len(easy_results) > 1:
                            # Sắp xếp theo vị trí Y
                            easy_results.sort(key=lambda x: x[0][0][1])
                            texts = [result[1] for result in easy_results]
                            text = '\n'.join(texts)
                            conf = sum([result[2] for result in easy_results]) / len(easy_results)
                        else:
                            best_result = max(easy_results, key=lambda x: x[2])
                            text = best_result[1]
                            conf = best_result[2]
                        
                        cleaned = clean_plate_text(text, is_multiline)
                        
                        if cleaned and conf > 0.6:
                            all_results.append(PlateResult(
                                text=cleaned,
                                confidence=conf,
                                method=f"EasyOCR_v{variant_idx}",
                                is_multiline=is_multiline
                            ))
                            
                except Exception as e:
                    print(f"[ERROR] EasyOCR variant {variant_idx} failed: {e}")
            
            # Method 3: Tesseract với PSM phù hợp
            try:
                if is_multiline:
                    # PSM 6 cho text blocks
                    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                else:
                    # PSM 8 cho single line
                    config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                
                text = pytesseract.image_to_string(processed_img, config=config).strip()
                
                if text:
                    cleaned = clean_plate_text(text, is_multiline)
                    if cleaned:
                        all_results.append(PlateResult(
                            text=cleaned,
                            confidence=0.75,
                            method=f"Tesseract_v{variant_idx}",
                            is_multiline=is_multiline
                        ))
                        
            except Exception as e:
                print(f"[ERROR] Tesseract variant {variant_idx} failed: {e}")
        
        # 5. Xử lý riêng từng vùng text nếu là multiline
        if is_multiline and len(text_regions) > 1:
            region_texts = []
            region_confidences = []
            
            for region_img, region_name in text_regions:
                best_region_result = None
                best_region_conf = 0
                
                # OCR từng vùng
                for processed_img in processed_variants[:3]:  # Chỉ dùng 3 variant tốt nhất
                    
                    # EasyOCR cho từng vùng
                    try:
                        easy_results = ocr_models.easy_ocr.readtext(
                            region_img,
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                            width_ths=0.8,
                            height_ths=0.8
                        )
                        
                        if easy_results:
                            best_result = max(easy_results, key=lambda x: x[2])
                            text = best_result[1]
                            conf = best_result[2]
                            
                            if conf > best_region_conf:
                                best_region_result = text
                                best_region_conf = conf
                                
                    except Exception as e:
                        print(f"[ERROR] Region OCR failed: {e}")
                
                if best_region_result:
                    cleaned_region = re.sub(r'[^A-Z0-9]', '', best_region_result.upper())
                    if cleaned_region:
                        region_texts.append(cleaned_region)
                        region_confidences.append(best_region_conf)
            
            # Ghép các vùng lại
            if len(region_texts) >= 2:
                combined_text = '-'.join(region_texts)
                avg_conf = sum(region_confidences) / len(region_confidences)
                
                all_results.append(PlateResult(
                    text=combined_text,
                    confidence=avg_conf,
                    method="RegionBased_OCR",
                    is_multiline=True
                ))
        
        # 6. Chọn kết quả tốt nhất
        if all_results:
            # Ưu tiên theo method và confidence
            priority_methods = ['PaddleOCR', 'RegionBased', 'EasyOCR', 'Tesseract']
            
            def get_method_priority(method_name):
                for i, priority in enumerate(priority_methods):
                    if priority in method_name:
                        return i
                return len(priority_methods)
            
            # Sắp xếp theo confidence và method priority
            all_results.sort(key=lambda x: (-x.confidence, get_method_priority(x.method)))
            
            best_result = all_results[0]
            print(f"[SUCCESS] Best result: {best_result.text} (conf: {best_result.confidence:.3f}, method: {best_result.method})")
            
            return best_result.text, best_result.confidence
        
        return None, 0.0
        
    except Exception as e:
        print(f"[FAILED] Enhanced multiline plate OCR failed: {e}")
        return None, 0.0

@timeout_with_executor(OCR_TIMEOUT)
def extract_text_container(image_crop: np.ndarray, ocr_models: OCRModels) -> Tuple[Optional[str], float]:
    """Enhanced OCR for container codes with vertical text support"""
    results = []
    
    try:
        # Preprocessing for container codes
        processed_images = []
        
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
        
        # Detect orientation
        orientation = detect_text_orientation(image_crop)
        
        # Resize if needed
        h, w = gray.shape
        if h < 80 or w < 200:
            scale = max(80/h, 200/w, 1.5)
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        
        # Multiple preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 8, 7, 21)
        processed_images.append(denoised)
        
        # For vertical text
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
        morph_v = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_v)
        processed_images.append(morph_v)
        
        # Rotation handling
        if orientation == "vertical":
            for angle in [90, -90, 180]:
                center = (w//2, h//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, rotation_matrix, (w, h))
                processed_images.append(rotated)
        
        # ===== OCR METHODS =====
        
        # 1. PaddleOCR
        if ocr_models.paddle_ocr is not None:
            for i, processed in enumerate(processed_images):
                try:
                    paddle_results = ocr_models.paddle_ocr.ocr(processed, cls=True)
                    if paddle_results and paddle_results[0]:
                        for line in paddle_results[0]:
                            text = line[1][0]
                            conf = line[1][1]
                            cleaned = clean_container_text(text)
                            if cleaned and conf > 0.75:
                                results.append((cleaned, conf, f"PaddleOCR_{i}"))
                except Exception as e:
                    print(f"[ERROR] PaddleOCR container method {i} failed: {e}")
        
        # 2. EasyOCR
        if ocr_models.easy_ocr is not None:
            for i, processed in enumerate(processed_images):
                try:
                    easy_results = ocr_models.easy_ocr.readtext(
                        processed,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        width_ths=0.8, height_ths=0.8
                    )
                    
                    if easy_results:
                        if orientation == "vertical":
                            combined_text = ''.join([r[1] for r in easy_results]) # type: ignore
                            cleaned = clean_container_text(combined_text)
                            if cleaned:
                                avg_conf = sum([r[2] for r in easy_results]) / len(easy_results) # type: ignore
                                results.append((cleaned, avg_conf, f"EasyOCR_Vertical_{i}"))
                        else:
                            best_result = max(easy_results, key=lambda x: x[2]) # type: ignore
                            text, conf = best_result[1], best_result[2] # type: ignore
                            cleaned = clean_container_text(text)
                            if cleaned and conf > 0.7: # type: ignore
                                results.append((cleaned, conf, f"EasyOCR_{i}"))
                                
                except Exception as e:
                    print(f"[ERROR] EasyOCR container method {i} failed: {e}")
        
        # 3. Tesseract
        for i, processed in enumerate(processed_images):
            try:
                text = pytesseract.image_to_string(processed, config=ocr_models.container_config).strip()
                if text:
                    cleaned = clean_container_text(text)
                    if cleaned:
                        results.append((cleaned, 0.75, f"Tesseract_{i}"))
            except Exception as e:
                print(f"[ERROR] Tesseract container method {i} failed: {e}")
        
        # Choose best result
        if results:
            priority_order = ['PaddleOCR', 'EasyOCR', 'Tesseract']
            
            def get_priority(method_name):
                for i, priority in enumerate(priority_order):
                    if priority in method_name:
                        return i
                return len(priority_order)
            
            results.sort(key=lambda x: (-x[1], get_priority(x[2])))
            best_result = results[0]
            return best_result[0], best_result[1]
        
        return None, 0.0
        
    except Exception as e:
        print(f"[FAILED] Enhanced container OCR failed: {e}")
        return None, 0.0

# ===== DETECTION FUNCTIONS =====
def detect_plates(frame: np.ndarray) -> List[Dict]:
    """Detect license plates in frame"""
    results = []
    try:
        yolo_out = model_manager.yolo_models['plate'](frame, conf=0.1, iou=0.5)[0]

        for box in yolo_out.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Bỏ qua detection confidence quá thấp
            if conf < 0.12:
                results.append({
                    "type": "plate",
                    "box": [x1, y1, x2, y2],
                    "text": None,
                    "confidence": conf
                })
                continue

            # Crop với padding phù hợp
            h, w = frame.shape[:2]
            padding = 8  # Tăng padding để capture đầy đủ text
            x1_c = max(0, x1 - padding)
            y1_c = max(0, y1 - padding)
            x2_c = min(w, x2 + padding)
            y2_c = min(h, y2 + padding)
            
            if x2_c <= x1_c or y2_c <= y1_c:
                continue
                
            cropped = frame[y1_c:y2_c, x1_c:x2_c]

            # Sử dụng OCR cải tiến
            text, ocr_conf = extract_text_plate(cropped, model_manager.ocr_models)
            
            results.append({
                "type": "plate",
                "box": [x1, y1, x2, y2],
                "text": text,
                "confidence": ocr_conf if text else conf,
                "detection_confidence": conf,
                "ocr_confidence": ocr_conf if text else 0.0
            })
    
    except Exception as e:
        print(f"[FAILED] Enhanced plate detection failed: {e}")
    
    return results

def detect_containers(frame: np.ndarray) -> List[Dict]:
    """Detect container codes in frame"""
    results = []
    try:
        yolo_out = model_manager.yolo_models['container'](frame, conf=0.15, iou=0.5)[0]
        
        for box in yolo_out.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            #  Detections without OCR
            if conf < 0.2:
                results.append({
                    "type": "container",
                    "box": [x1, y1, x2, y2],
                    "text": None,
                    "confidence": conf
                })
                continue

            # Crop with padding
            h, w = frame.shape[:2]
            padding = 8
            x1_c = max(0, x1 - padding)
            y1_c = max(0, y1 - padding)
            x2_c = min(w, x2 + padding)
            y2_c = min(h, y2 + padding)
            
            if x2_c <= x1_c or y2_c <= y1_c:
                continue
                
            cropped = frame[y1_c:y2_c, x1_c:x2_c]
            text, ocr_conf = extract_text_container(cropped, model_manager.ocr_models) # type: ignore
            
            results.append({
                "type": "container",
                "box": [x1, y1, x2, y2],
                "text": text,
                "confidence": ocr_conf if text else conf
            })
    
    except Exception as e:
        print(f"[ERROR] Container detection failed: {e}")
    
    return results

def detect_faces(frame):
    """Detect and recognize faces in frame"""
    if model_manager.face_classifier is None or model_manager.label_encoder is None or model_manager.face_models.get('mtcnn') is None:
        return []

    results = []
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mtcnn = model_manager.face_models['mtcnn']
        boxes, probs = mtcnn.detect(img_rgb) # type: ignore

        if boxes is None or len(boxes) == 0:
            return []

        for i, box in enumerate(boxes):
            prob = float(probs[i]) if probs is not None else 0.0
            if prob < FACE_CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            h, w = img_rgb.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            
            if x2c <= x1c or y2c <= y1c:
                continue

            # Extract face and get embedding
            face_crop = img_rgb[y1c:y2c, x1c:x2c]
            face_pil = Image.fromarray(face_crop)
            face_tensor = face_transform(face_pil).unsqueeze(0).to(device) # type: ignore

            with torch.no_grad():
                embedding = model_manager.face_models['facenet'](face_tensor).cpu().numpy()

            # Classify face
            proba_list = model_manager.face_classifier.predict_proba(embedding)[0]
            best_idx = np.argmax(proba_list)
            best_prob = float(proba_list[best_idx])
            
            name = model_manager.label_encoder.inverse_transform([best_idx])[0] if best_prob >= FACE_CONFIDENCE_THRESHOLD else None

            results.append({
                "type": "face",
                "box": [x1c, y1c, x2c, y2c],
                "text": name,
                "confidence": best_prob
            })

        return results

    except Exception as e:
        print(f"[FAILED] Face detection failed: {e}")
        return []

# ===== DRAWING AND VISUALIZATION =====
def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    colors = {
        "plate": (0, 255, 0),      # Green
        "container": (255, 0, 0),   # Blue  
        "face": (0, 0, 255),       # Red
        "seal": (255, 255, 0)      # Cyan
    }
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        det_type = det["type"]
        text = det["text"] if det["text"] is not None else "None"
        conf = det["confidence"]
        color = colors.get(det_type, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{det_type}: {text} ({conf:.2f})"
        cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# ===== MAIN PROCESSING FUNCTIONS =====
@timeout_with_executor(IMAGE_PROCESSING_TIMEOUT)
def process_image(frame):
    """Main image processing function with timeout protection"""
    detections = []
    try:
        # Sequential detection to ensure OCR completes before moving on
        plate_detections = detect_plates(frame)
        detections.extend(plate_detections)

        container_detections = detect_containers(frame)
        detections.extend(container_detections)

        face_detections = detect_faces(frame)
        detections.extend(face_detections)

        # Draw annotations
        annotated = draw_detections(frame.copy(), detections)
        return annotated, detections

    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        return frame, []

def extract_metadata(detections: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract metadata from detections"""
    metadata = {"plate": "None", "container": "None", "face": "None", "seal": "None"}
    
    for det in detections:
        det_type = det["type"]
        text = det["text"]
        
        if text is not None and text != "None":
            metadata[det_type] = text
    
    return metadata

# ===== STREAMING FUNCTIONS =====
def update_latest_frame_and_metadata(frame, metadata):
    """Update global frame and metadata for streaming"""
    with global_state.frame_lock:
        global_state.latest_frame = frame.copy()
        global_state.latest_metadata = metadata.copy()  

def generate_mjpeg():
    """Generate MJPEG stream for video feed"""
    while True:
        with global_state.frame_lock:
            if global_state.latest_frame is not None:
                current_frame = global_state.latest_frame.copy() 
            else:
                current_frame = None
        
        if current_frame is not None:
            _, jpeg = cv2.imencode('.jpg', current_frame)
            frame = jpeg.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
        else:
            time.sleep(0.05)

# # ===== API ENDPOINTS =====
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": device,
        "models_loaded": global_state.models_loaded,
        "models": {
            "yolo_plate": model_manager.yolo_models.get('plate') is not None,
            "yolo_container": model_manager.yolo_models.get('container') is not None,
            "mtcnn": model_manager.face_models.get('mtcnn') is not None,
            "facenet": model_manager.face_models.get('facenet') is not None,
            "face_classifier": model_manager.face_classifier is not None,
            "ocr_models": model_manager.ocr_models is not None,
        }
    }

@app.get("/start-stream")
def start_camera_stream():
    """Start camera client stream"""
    try:
        subprocess.Popen(["python", "client_camera.py"])
        return JSONResponse({"message": "Client camera started"}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Cannot start client_camera: {e}"}, status_code=500)

@app.get("/video-feed/combined-detection")
def get_video_feed():
    """Get video feed with combined detection"""
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/combined-detection")
async def websocket_combined_detection(websocket: WebSocket):
    """Main WebSocket endpoint for combined detection"""
    await websocket.accept()
    print("[WebSocket] Client connected")
    
    try:
        while True:
            # Receive message with timeout
            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=WEBSOCKET_TIMEOUT
                )
            except asyncio.TimeoutError:
                print("[WebSocket] Receive timeout, sending ping")
                await websocket.ping() # type: ignore
                continue
            
            try:
                message = json.loads(raw_message)
                frame_b64 = message.get("image")
                
                if not frame_b64:
                    await websocket.send_text(json.dumps({
                        "success": False,
                        "error": "No image data received"
                    }))
                    continue
                
                # Decode image
                img_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_text(json.dumps({
                        "success": False,
                        "error": "Cannot decode image"
                    }))
                    continue
                
                # Process image with timeout protection
                start_time = time.time()
                result = process_image(frame)
                
                if result is None:
                    await websocket.send_text(json.dumps({
                        "success": False,
                        "error": "Processing timeout"
                    }))
                    continue
                
                annotated_frame, detections = result
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Extract metadata and update global state
                metadata = extract_metadata(detections)
                update_latest_frame_and_metadata(annotated_frame, metadata)
                
                # Encode response
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                annotated_b64 = base64.b64encode(buffer).decode('utf-8') # type: ignore
                
                response = {
                    "success": True,
                    "image": annotated_b64,
                    "detections": detections,
                    "metadata": metadata,
                    "processing_time_ms": round(processing_time, 2)
                }
                
                await websocket.send_text(json.dumps(response))
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "success": False,
                    "error": "Invalid JSON format"
                }))
            except Exception as e:
                print(f"[WebSocket Error] Processing frame: {e}")
                await websocket.send_text(json.dumps({
                    "success": False,
                    "error": f"Processing error: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket Error] Connection error: {e}")

@app.websocket("/ws/flutter-metadata")
async def websocket_flutter_metadata(websocket: WebSocket):
    """WebSocket endpoint for Flutter app metadata only"""
    await websocket.accept()
    print("[Flutter WebSocket] Client connected")
    
    try:
        last_metadata = None
        while True:
            with global_state.frame_lock:
                current_metadata = global_state.latest_metadata.copy()  # Sửa ở đây
            
            # Only send if metadata has changed
            if current_metadata != last_metadata:
                await websocket.send_json(current_metadata)
                last_metadata = current_metadata.copy()
            
            await asyncio.sleep(0.1)  # 100ms polling interval
            
    except WebSocketDisconnect:
        print("[Flutter WebSocket] Client disconnected")
    except Exception as e:
        print(f"[Flutter WebSocket Error] {e}")