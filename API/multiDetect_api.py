import subprocess
import time
import uuid
import cv2
from paddleocr import PaddleOCR
import torch
import json
import base64
import numpy as np
import joblib
import re
import easyocr
import threading
import concurrent.futures
from logging import debug
from fastapi import Request
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
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GlobalState:
    def __init__(self):
        self.latest_frame = None
        self.latest_metadata = {"plate": "None", "container": "None", "face": "None", "seal": "None"}
        self.frame_lock = threading.Lock()
        self.models_loaded = False

# Detection thresholds
FACE_CONFIDENCE_THRESHOLD = 0.3  # Face detection threshold
OCR_CONFIDENCE_THRESHOLD = 0.3    # OCR confidence threshold
SEAL_OCR_CONFIDENCE_THRESHOLD = 0.25  # Threshold for seal detection

# WebSocket timeout settings
WEBSOCKET_TIMEOUT = 60  # seconds
PING_INTERVAL = 10      # seconds
PING_TIMEOUT = 5        # seconds

# Processing timeout settings
IMAGE_PROCESSING_TIMEOUT = 45  # seconds 
OCR_TIMEOUT = 60  # seconds per OCR operation

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
            self.yolo_models['plate'] = YOLO("modelAI/detect_PlateNumber.pt").to(device_cpu)
            self.yolo_models['container'] = YOLO("modelAI/detect_ContainerCode.pt").to(device_cpu)

            self.face_models['mtcnn'] = MTCNN(keep_all=True, device=device_gpu)
            self.face_models['facenet'] = InceptionResnetV1(pretrained='vggface2').eval().to(device_gpu)

            try: 
                self.face_classifier = joblib.load('modelAI/face_classifier.joblib')
                self.label_encoder = joblib.load('modelAI/label_encoder.joblib')
                print("[INFO] Face classifier loaded successfully")
            except Exception as e:
                print(f"[WARNING] Face classifier not found: {e}")
                self.face_classifier = None
                self.label_encoder = None

            try:
                self.yolo_models['container'] = YOLO("modelAI/detect_ContainerCode.pt")
                print("[INFO] Container model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load container model: {e}")
                self.yolo_models['container'] = None  

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
                lang="en"
            )
            print("[INFO] PaddleOCR initialized successfully")

        except Exception as e:
            print(f"[ERROR] PaddleOCR initialization failed: {e}")
            self.paddle_ocr = None

        self.easy_ocr = easyocr.Reader(['en'], gpu=True)


@dataclass
class PlateResult:
    text: str
    confidence: float
    method: str
    is_multiline: bool

@dataclass
class ContainerResult:
    text: str
    confidence: float
    method: str

# ===== UTILITY FUNCTIONS =====
global_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
def timeout_with_executor(timeout_sec: float):
    def decorator(func):
        def wrapper(*args, **kwargs):
            future = global_executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                print(f"[TIMEOUT] {func.__name__} timed out after {timeout_sec}s")
                return None
        return wrapper
    return decorator

# ===== TEXT PROCESSING FUNCTIONS =====
def create_rotation_variants(image: np.ndarray) -> List[np.ndarray]:
    """
    Tạo rotation variants cho container text
    """
    variants = [image]
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        center = (w//2, h//2)
        
        # Chỉ tạo rotation nếu cần thiết
        orientation = detect_text_orientation(image)
        
        if orientation == "vertical":
            for angle in [90, -90]:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, M, (w, h), 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=255) # type: ignore
                variants.append(rotated)
        
        return variants
        
    except Exception as e:
        print(f"[ERROR] Rotation variants failed: {e}")
        return [image]
    
def detect_text_orientation(image: np.ndarray) -> str:
    """Phát hiện hướng của text (horizontal/vertical)"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # Tính gradient theo 2 hướng
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tính strength của edges
        horizontal_strength = np.mean(np.abs(grad_x))
        vertical_strength = np.mean(np.abs(grad_y))
        
        # Aspect ratio cũng là indicator
        aspect_ratio = w / h
        
        # Kết hợp cả hai yếu tố
        if vertical_strength > horizontal_strength * 1.2 or aspect_ratio < 0.5:
            return "vertical"
        else:
            return "horizontal"
            
    except Exception as e:
        print(f"[ERROR] Orientation detection failed: {e}")
        return "horizontal"

# ===== PREPROCESS IMAGE FOR OCR FUNCTIONS =====
def preprocess_for_plate_text(image: np.ndarray) -> List[np.ndarray]:
    """Tiền xử lý cho biển số xe 2 dòng"""
    processed_variants = []
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        if h < 60 or w < 200:
            scale_factor = max(60/h, 200/w, 1.5)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = new_h, new_w
        
        # 1. Baseline - CLAHE moderate để cải thiện contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 4))
        enhanced = clahe.apply(gray)
        processed_variants.append(enhanced)
        
        # 2. Giảm noise + tăng contrast - phù hợp với ảnh có nhiễu
        # Gaussian blur nhẹ trước khi CLAHE
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 2))
        enhanced_strong = clahe_strong.apply(blurred)
        processed_variants.append(enhanced_strong)
        
        # 3. Xử lý cho ảnh tối/thiếu sáng (như ảnh 2)
        # Gamma correction + CLAHE
        gamma = 1.5  # Tăng độ sáng
        gamma_corrected = np.power(gray / 255.0, 1.0/gamma) # type: ignore
        gamma_corrected = np.uint8(gamma_corrected * 255)
        enhanced_gamma = clahe.apply(gamma_corrected) # type: ignore
        processed_variants.append(enhanced_gamma)
        
        # 4. Sharpening cho text mờ
        kernel_sharpen = np.array([[-1, -1, -1], 
                                  [-1, 9, -1], 
                                  [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        # Clamp values
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        processed_variants.append(sharpened)
        
        # 5. Morphological operations để kết nối ký tự bị gãy
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_close)
        processed_variants.append(closed)
        
        # 6. Adaptive threshold cho các trường hợp khó
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 8
        )
        processed_variants.append(adaptive_thresh)
        
        return processed_variants[:6]
    
    except Exception as e:
        print(f"[ERROR] Plate preprocessing failed: {e}")
        return [gray] if 'gray' in locals() else []

def preprocess_for_container_text(image: np.ndarray):
    "Tiền xử lý ảnh cho OCR số container"
    
    processed_variants = []
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        # Tăng kích thước tối thiểu để OCR đọc rõ hơn
        min_width, min_height = 400, 150  
        if h < min_height or w < min_width:
            scale = max(min_width / w, min_height / h, 2.5)  # Tăng scale factor
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
            h, w = gray.shape

        # 1. Baseline với CLAHE mạnh hơn
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 2))  # Tăng clipLimit
        enhanced = clahe.apply(gray)
        processed_variants.append(enhanced)
        
        # 2. Bilateral filter + contrast enhancement
        bilateral = cv2.bilateralFilter(gray, 9, 80, 80)
        # Tăng contrast thêm bằng histogram stretching
        p2, p98 = np.percentile(bilateral, (2, 98))
        bilateral_stretched = np.clip((bilateral - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        enhanced_bilateral = clahe.apply(bilateral_stretched)
        processed_variants.append(enhanced_bilateral)
        
        # 3. Morphological operations để làm rõ text
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Opening để loại bỏ noise
        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_rect)
        # Closing để kết nối text bị gãy
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_rect)
        processed_variants.append(closed)
        
        # 4. Unsharp masking mạnh hơn
        gaussian = cv2.GaussianBlur(enhanced, (3, 3), 0)
        unsharp = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)  # Tăng weight
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        processed_variants.append(unsharp)
        
        # 5. Adaptive threshold với nhiều variant
        # Variant 1: Gaussian adaptive
        adaptive1 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        processed_variants.append(adaptive1)
        
        # Variant 2: Mean adaptive
        adaptive2 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 19, 10
        )
        processed_variants.append(adaptive2)
        
        return processed_variants
    except Exception as e:
        print(f"[ERROR] Container preprocessing failed: {e}")
        return [gray] if 'gray' in locals() else []
    
def detect_plate_layout(image: np.ndarray) -> Dict[str, any]: # type: ignore
    """Phát hiện layout của biển số xe (1 dòng hay 2 dòng)"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # Aspect ratio analysis
        aspect_ratio = w / h
        
        # Contour analysis để detect text regions
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc contours có thể là text
        text_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Lọc noise
                x, y, w_c, h_c = cv2.boundingRect(contour)
                if w_c > 10 and h_c > 10:  # Kích thước hợp lý
                    text_contours.append((x, y, w_c, h_c))
        
        # Phân tích vị trí các text regions
        if len(text_contours) >= 2:
            # Sắp xếp theo y
            text_contours.sort(key=lambda x: x[1])
            
            # Kiểm tra có text ở nhiều dòng không
            y_positions = [tc[1] for tc in text_contours]
            y_diff = max(y_positions) - min(y_positions)
            
            # Nếu khoảng cách Y lớn => multiline
            is_multiline = y_diff > h * 0.3
        else:
            # Fallback: dựa vào aspect ratio
            is_multiline = aspect_ratio < 3.0
        
        return {
            'is_multiline': is_multiline,
            'reading_direction': 'left_to_right',
            'aspect_ratio': aspect_ratio,
            'confidence': 0.8 if len(text_contours) >= 2 else 0.6
        }
        
    except Exception as e:
        print(f"[ERROR] Layout detection failed: {e}")
        return {
            'is_multiline': False,
            'reading_direction': 'left_to_right',
            'aspect_ratio': 1.0,
            'confidence': 0.3
        }    

def format_vietnam_plate(text: str) -> Optional[str]:
    """
    Format theo chuẩn biển số Việt Nam: 2 số + 1-2 chữ + 4-6 số
    """
    if not text or len(text) < 6:
        return None
    
    # Mapping để sửa lỗi OCR
    to_digit = {'O': '0', 'D': '0', 'Q': '0', 'B': '8', 'S': '5', 'Z': '2', 'G': '6', 'I': '1', 'L': '1'}
    to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
    
    chars = list(text)
    
    # 2 ký tự đầu phải là số (mã tỉnh)
    for i in range(min(2, len(chars))):
        if chars[i] in to_digit:
            chars[i] = to_digit[chars[i]]
    
    # Ký tự thứ 3 phải là chữ (loại xe)
    if len(chars) >= 3 and chars[2] in to_letter:
        chars[2] = to_letter[chars[2]]
    
    # Ký tự thứ 4 có thể là chữ (trong trường hợp 2 chữ)
    if len(chars) >= 4 and not chars[3].isdigit():
        if chars[3] in to_letter:
            chars[3] = to_letter[chars[3]]
        # Nếu không phải chữ hợp lệ, coi như số
        elif chars[3] in to_digit:
            chars[3] = to_digit[chars[3]]
    
    # Các ký tự còn lại phải là số
    start_idx = 4 if len(chars) >= 4 and chars[3].isalpha() else 3
    for i in range(start_idx, len(chars)):
        if chars[i] in to_digit:
            chars[i] = to_digit[chars[i]]
    
    result = ''.join(chars)
    
    # Kiểm tra format hợp lệ
    if re.match(r'^\d{2}[A-Z]\d{4,6}$', result) or re.match(r'^\d{2}[A-Z]{2}\d{4,6}$', result):
        return result
    
    return result if 6 <= len(result) <= 10 else None

def clean_plate_text(text: str, is_multiline: bool = False) -> Optional[str]:
    """Cải tiến text cleaning cho biển số xe"""
    if not text:
        return None
    
    # Loại bỏ ký tự đặc biệt và chuẩn hóa
    text = re.sub(r'[^\w]', '', text.upper().strip())
    
    # Sửa lỗi OCR
    ocr_corrections = {
        # Số vs chữ
        'O': '0', 'I': '1', 'l': '1', '|': '1', 'L': '1',
        'S': '5', 'Z': '2', 'G': '6', 'B': '8', 'Q': '0', 'D': '0',
        # Ký tự đặc biệt
        '@': '8', '?': '7', '%': '8', '&': '8',
        # Khoảng trắng thừa
        ' ': '', '\t': '', '\n': ' ' if is_multiline else ''
    }
    
    for wrong, correct in ocr_corrections.items():
        text = text.replace(wrong, correct)

    # Xử lý multiline
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
    
    # Áp dụng format chuẩn biển số Việt Nam
    return format_vietnam_plate(combined)


def clean_container_text(text: str) -> Optional[str]:
    """Xử lý text container code dọc"""
    if not text:
        return None
        
    # Container format: 4 letters + 7 digits (e.g., ABCD1234567)
    text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
    
    # OCR corrections mở rộng
    corrections = {
        '0': 'O', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 
        'G': '6', 'B': '8', 'Q': '0', 'D': '0',
        '@': '8', '?': '7', '%': '8'
    }

    # # Tách chữ và số
    # letters = ''.join([c for c in text if c.isalpha()])
    # numbers = ''.join([c for c in text if c.isdigit()])
    
    # # Format chuẩn container: 4 chữ + 7 số
    # if len(letters) >= 4 and len(numbers) >= 6:
    #     # 4 ký tự đầu là chữ
    #     letter_part = letters[:4]
    #     # Sửa lỗi OCR cho phần chữ
    #     letter_corrected = ''
    #     for char in letter_part:
    #         if char.isdigit():
    #             # Chuyển số thành chữ
    #             digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
    #             letter_corrected += digit_to_letter.get(char, char)
    #         else:
    #             letter_corrected += char
        
    #     # Phần số (lấy 6-7 ký tự)
    #     number_part = numbers[:7] if len(numbers) >= 7 else numbers
    #     # Sửa lỗi OCR cho phần số
    #     number_corrected = ''
    #     for char in number_part:
    #         if char.isalpha():
    #             number_corrected += corrections.get(char, char)
    #         else:
    #             number_corrected += char
        
    #     result = letter_corrected + number_corrected
    #     result = result[:10]

    #     # Kiểm tra format cuối cùng
    #     if len(result) >= 10 and re.match(r'^[A-Z]{4}\d{6,7}$', result):
    #         return result
        
    # text = text[:10]
    # return text if len(text) >= 6 else None

    if len(text) >= 10:
        # 4 ký tự đầu phải là chữ
        letter_part = text[:4]
        number_part = text[4:]
        
        # Sửa lỗi OCR cho phần chữ
        corrected_letters = ''
        for char in letter_part:
            if char.isdigit():
                digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
                corrected_letters += digit_to_letter.get(char, char)
            else:
                corrected_letters += char
        
        # Sửa lỗi OCR cho phần số (lấy 6-7 ký tự)
        corrected_numbers = ''
        for char in number_part[:7]:
            if char.isalpha():
                corrected_numbers += corrections.get(char, '0')
            else:
                corrected_numbers += char
        
        result = corrected_letters + corrected_numbers
        
        # Kiểm tra format cuối cùng
        if re.match(r'^[A-Z]{4}\d{6,7}$', result):
            return result
    
    # Fallback: làm sạch basic
    if len(text) >= 8:
        return text[:11]  # Giới hạn độ dài
    
    return None

@timeout_with_executor(OCR_TIMEOUT)
def extract_text_plate(image_crop: np.ndarray, ocr_models) -> Tuple[Optional[str], float]:
    """Cải tiến OCR cho biển số xe 2 dòng"""
    all_results = []
    
    try:
        # 1. Phát hiện layout
        layout_info = detect_plate_layout(image_crop)
        is_multiline = layout_info['is_multiline']
        reading_direction = layout_info.get('reading_direction', 'left_to_right')
        print(f"[INFO] Detected layout: {'Multiline' if is_multiline else 'Single line'}")
        
        # 2. Tiền xử lý chuyên biệt
        processed_variants = preprocess_for_plate_text(image_crop)
        
        # 3. OCR với nhiều phương pháp và scoring
        for variant_idx, processed_img in enumerate(processed_variants[:4]):
            # Fallback OCR strategy
            ocr_attempts = []

            # PaddleOCR
            try:
                if ocr_models.paddle_ocr:
                    if len(processed_img.shape) == 2:
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                    elif processed_img.shape[2] == 1:
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

                    paddle_results = ocr_models.paddle_ocr.ocr(processed_img)
                    
                    # Xử lý nhiều trường hợp trả về khác nhau
                    texts = []
                    confidences = []
                    text_boxes = []
                    
                    if paddle_results is not None:
                        # Trường hợp 1: paddle_results là list chứa list
                        if isinstance(paddle_results, list) and len(paddle_results) > 0:
                            first_result = paddle_results[0]
                            
                            # Trường hợp 1a: first_result là list các detection
                            if isinstance(first_result, list):
                                for line in first_result:
                                    try:
                                        # Kiểm tra cấu trúc: [bbox, (text, confidence)]
                                        if isinstance(line, list) and len(line) >= 2:
                                            bbox = line[0]
                                            text_info = line[1]
                                            if isinstance(text_info, tuple) and len(text_info) >= 2:
                                                text, conf = text_info[0], text_info[1]
                                                if text and isinstance(text, str) and text.strip():
                                                    # Tính toán vị trí trung tâm của bbox
                                                    center_y = sum([p[1] for p in bbox]) / 4
                                                    center_x = sum([p[0] for p in bbox]) / 4
                                                    
                                                    texts.append(text.strip())
                                                    confidences.append(float(conf))
                                                    text_boxes.append({'center_x': center_x, 'center_y': center_y})
                                    except (IndexError, TypeError, ValueError) as line_e:
                                        print(f"[WARN] Skipping malformed PaddleOCR line in variant {variant_idx}: {line_e}")
                                        continue
                                if texts and text_boxes:
                                    # Kết hợp texts với positions
                                    combined_data = list(zip(texts, confidences, text_boxes))
                                    
                                    if is_multiline:
                                        # Sắp xếp theo Y trước (top to bottom), rồi X (left to right)
                                        combined_data.sort(key=lambda x: (x[2]['center_y'], x[2]['center_x']))
                                    else:
                                        # Sắp xếp theo X (left to right)
                                        combined_data.sort(key=lambda x: x[2]['center_x'])
                                    
                                    # Tách lại thành các list riêng biệt
                                    texts = [item[0] for item in combined_data]
                                    confidences = [item[1] for item in combined_data]

                                    if is_multiline and len(texts) > 1:
                                        combined_text = '\n'.join(texts)
                                    else:
                                        combined_text = ''.join(texts)
                    
                            
                            # Trường hợp 1b: first_result là tuple (text, confidence)
                            elif isinstance(first_result, tuple) and len(first_result) >= 2:
                                text, conf = first_result[0], first_result[1]
                                if text and isinstance(text, str) and text.strip():
                                    texts.append(text.strip())
                                    confidences.append(float(conf))
                            
                            # Trường hợp 1c: first_result là string
                            elif isinstance(first_result, str) and first_result.strip():
                                texts.append(first_result.strip())
                                confidences.append(0.5)  # Default confidence
                            
                            # Trường hợp 1d: first_result là dict
                            elif isinstance(first_result, dict) and 'rec_texts' in first_result and 'rec_scores' in first_result:
                                try:
                                    if 'rec_texts' in first_result and 'rec_scores' in first_result:
                                        texts = [t.strip() for t in first_result['rec_texts'] if isinstance(t, str) and t.strip()]
                                        confidences = [float(s) for s in first_result['rec_scores']]
                                        if texts and confidences:
                                            combined_text = ''.join(texts)
                                            avg_conf = sum(confidences) / len(confidences)
                                            cleaned = clean_container_text(combined_text)
                                            if cleaned:
                                                ocr_attempts.append((cleaned, avg_conf, f"PaddleOCR_v{variant_idx}"))
                                                print(f"[SUCCESS] PaddleOCR container variant {variant_idx}: {cleaned} (conf: {avg_conf:.3f})")
                                except Exception as parse_e:
                                    print(f"[WARN] Failed to parse dict result in PaddleOCR container: {parse_e}")
                        
                        # Trường hợp 2: paddle_results trực tiếp là list các detection
                        elif isinstance(paddle_results, list):
                            for item in paddle_results:
                                try:
                                    if isinstance(item, list) and len(item) >= 2:
                                        text_info = item[1]
                                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                                            text, conf = text_info[0], text_info[1]
                                            if text and isinstance(text, str) and text.strip():
                                                texts.append(text.strip())
                                                confidences.append(float(conf))
                                except (IndexError, TypeError, ValueError):
                                    continue
                    
                    # Xử lý kết quả thu được
                    if texts and confidences:
                        if is_multiline and len(texts) > 1:
                            combined_text = '\n'.join(texts)
                        else:
                            combined_text = texts[0] if texts else ""
                        
                        avg_conf = sum(confidences) / len(confidences)
                        cleaned = clean_plate_text(combined_text, is_multiline)
                        
                        if cleaned:
                            ocr_attempts.append((cleaned, avg_conf, f"PaddleOCR_v{variant_idx}"))
                            print(f"[SUCCESS] PaddleOCR variant {variant_idx}: {cleaned} (conf: {avg_conf:.3f})")
                    else:
                        print(f"[INFO] PaddleOCR variant {variant_idx} returned no valid text")
                        
            except Exception as e:
                print(f"[ERROR] PaddleOCR variant {variant_idx} failed: {e}")


            # Try EasyOCR
            try:
                if ocr_models.easy_ocr:
                    easy_results = ocr_models.easy_ocr.readtext(
                        processed_img,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                        width_ths=0.6,
                        height_ths=0.3 if is_multiline else 0.6,
                        paragraph=is_multiline,
                        decoder='greedy'
                    )
                    if easy_results and isinstance(easy_results, list):
                        valid_items = [r for r in easy_results if isinstance(r, (list, tuple)) and len(r) >= 3]
                        
                        if valid_items:
                            # IMPROVED: Sắp xếp theo vị trí
                            if is_multiline:
                                # Sắp xếp theo Y trước, rồi X
                                valid_items.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
                            else:
                                # Sắp xếp theo X
                                valid_items.sort(key=lambda x: x[0][0][0])
                            
                            if is_multiline and len(valid_items) > 1:
                                texts = [r[1] for r in valid_items]
                                combined_text = '\n'.join(texts)
                                avg_conf = sum([r[2] for r in valid_items]) / len(valid_items)
                                cleaned = clean_plate_text(combined_text, is_multiline)
                                if cleaned:
                                    ocr_attempts.append((cleaned, avg_conf, f"EasyOCR_v{variant_idx}"))
                                
                            else:
                                best_result = max(valid_items, key=lambda x: x[2])
                                combined_text = best_result[1]
                                avg_conf = best_result[2]
                                cleaned = clean_plate_text(combined_text, is_multiline)
                                if cleaned:
                                    ocr_attempts.append((cleaned, avg_conf, f"EasyOCR_v{variant_idx}"))
                        
                        else:
                            print(f"[INFO] EasyOCR variant {variant_idx} returned empty or invalid format")

            except Exception as e:
                print(f"[ERROR] EasyOCR variant {variant_idx} failed: {e}")

            # Append best attempt
            if ocr_attempts:
                best = max(ocr_attempts, key=lambda x: x[1])
                if best[1] > 0.3:
                    bonus = 0.1 if variant_idx < 4 else 0
                    final_conf = min(best[1] + bonus, 1.0)
                    all_results.append(PlateResult(
                        text=best[0],
                        confidence=final_conf,
                        method=best[2],
                        is_multiline=is_multiline
                    ))

        if all_results:
            # Voting
            if len(all_results) >= 3:
                votes = {}
                for res in all_results:
                    if res.text in votes:
                        votes[res.text]['count'] += 1
                        votes[res.text]['conf'] += res.confidence
                    else:
                        votes[res.text] = {'count': 1, 'conf': res.confidence}

                for t, v in votes.items():
                    if v['count'] >= 2:
                        avg_conf = v['conf'] / v['count']
                        bonus = 0.1 * (v['count'] - 1)
                        all_results.append(PlateResult(
                            text=t,
                            confidence=min(avg_conf + bonus, 1.0),
                            method=f"Consensus_{v['count']}",
                            is_multiline=is_multiline
                        ))

            # Score & chọn kết quả tốt nhất
            def score(r):
                base = r.confidence
                method_bonus = 0.1 if 'PaddleOCR' in r.method else 0.05 if 'EasyOCR' in r.method else 0
                multiline_bonus = 0.05 if r.is_multiline else 0
                
                # Pattern validation bonus
                text_clean = r.text.replace('-', '').replace('\n', '')
                length_bonus = 0.05 if 6 <= len(text_clean) <= 10 else -0.1
                
                # Vietnam license plate pattern bonus
                vietnam_pattern_bonus = 0
                if re.match(r'^\d{2}[A-Z]\d{4,6}$', text_clean):  # 12A34567
                    vietnam_pattern_bonus = 0.1
                elif re.match(r'^\d{2}[A-Z]{2}\d{4,6}$', text_clean):  # 12AB34567
                    vietnam_pattern_bonus = 0.1
                elif re.match(r'^\d{2}[A-Z]\d{3}\.\d{2}$', text_clean):  # 12A123.45
                    vietnam_pattern_bonus = 0.1
                
                return base + method_bonus + multiline_bonus + length_bonus + vietnam_pattern_bonus
            
            all_results.sort(key=score, reverse=True)
            best_result = all_results[0]
            print(f"[SUCCESS] Best result: {best_result.text} (conf: {best_result.confidence:.3f}, method: {best_result.method})")
            return best_result.text, best_result.confidence

        return None, 0.0

    except Exception as e:
        print(f"[FAILED] Enhanced plate OCR failed: {e}")
        return None, 0.0

    
@timeout_with_executor(OCR_TIMEOUT)
def extract_text_container(image_crop: np.ndarray, ocr_models: OCRModels) -> Tuple[Optional[str], float]:
    """Enhanced OCR for container codes with vertical text support"""
    all_results = []
    
    try:
        # Preprocessing for container codes
        processed_images = preprocess_for_container_text(image_crop)
        
        # Detect orientation
        orientation = detect_text_orientation(image_crop)
        
        # Thêm rotation variants nếu cần
        if orientation == "vertical":
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
            h, w = gray.shape
            
            for angle in [90, -90, 180]:  # Thêm 180 độ
                center = (w//2, h//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Tính toán kích thước mới sau rotation
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                new_w = int((h * sin_angle) + (w * cos_angle))
                new_h = int((h * cos_angle) + (w * sin_angle))
                
                # Điều chỉnh translation
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                rotated = cv2.warpAffine(gray, rotation_matrix, (new_w, new_h), borderValue=255)
                processed_images.append(rotated)
        
        # ===== OCR METHODS =====
        # Fallback OCR
        for variant_idx, processed_img in enumerate(processed_images[:6]):
            ocr_attempts = []

            # PaddleOCR
            if ocr_models.paddle_ocr:
                try:
                    if len(processed_img.shape) == 2:
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

                    paddle_res = ocr_models.paddle_ocr.ocr(
                        processed_img,
                        cls=True,  # Bật text direction classification
                        rec=True,  # Bật text recognition
                        det=True   # Bật text detection
                        )
                    
                    texts, confs = [], []

                    if isinstance(paddle_res, list) and len(paddle_res) > 0:
                        first = paddle_res[0]

                        # Trường hợp list các dòng text
                        if isinstance(first, list):
                            # Sắp xếp theo vị trí (quan trọng cho container code)
                            try:
                                if orientation == "vertical":
                                    # Sắp xếp từ trên xuống dưới, trái sang phải
                                    first = sorted(first, key=lambda x: (
                                        min([p[1] for p in x[0]]),  # Y coordinate (top)
                                        min([p[0] for p in x[0]])   # X coordinate (left)
                                    ))
                                else:
                                    # Sắp xếp từ trái sang phải, trên xuống dưới
                                    first = sorted(first, key=lambda x: (
                                        min([p[0] for p in x[0]]),  # X coordinate
                                        min([p[1] for p in x[0]])   # Y coordinate
                                    ))
                            except Exception:
                                pass

                            for line_idx, line in enumerate(first):
                                try:
                                    if isinstance(line, list) and len(line) >= 2:
                                        text_info = line[1]
                                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                                            text, conf = text_info[0], text_info[1]
                                            if text and isinstance(text, str) and len(text.strip()) > 0:
                                                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
                                                if len(cleaned_text) >= 3:  # Lọc text quá ngắn
                                                    texts.append(cleaned_text)
                                                    confs.append(float(conf))
                                                    print(f"[DEBUG] PaddleOCR line {line_idx}: '{cleaned_text}' (conf: {conf:.3f})")
                                        elif isinstance(text_info, str) and len(text_info.strip()) > 0:
                                            cleaned_text = re.sub(r'[^A-Z0-9]', '', text_info.upper().strip())
                                            if len(cleaned_text) >= 3:
                                                texts.append(cleaned_text)
                                                confs.append(0.5)
                                                print(f"[DEBUG] PaddleOCR line {line_idx}: '{cleaned_text}' (default conf)")
                                except Exception as e:
                                    print(f"[WARN] Skipping malformed line {line_idx}: {e}")

                    if texts and confs:
                        # Thử nhiều cách kết hợp text
                        combinations = []
                        
                        # 1. Kết hợp tất cả text
                        combined_all = ''.join(texts)
                        if len(combined_all) >= 6:
                            combinations.append((combined_all, sum(confs) / len(confs)))
                        
                        # 2. Lấy text có confidence cao nhất
                        if texts:
                            max_conf_idx = confs.index(max(confs))
                            best_text = texts[max_conf_idx]
                            if len(best_text) >= 6:
                                combinations.append((best_text, confs[max_conf_idx]))
                        
                        # 3. Kết hợp 2 text đầu tiên (thường là code chính)
                        if len(texts) >= 2:
                            combined_two = texts[0] + texts[1]
                            if len(combined_two) >= 6:
                                avg_conf = (confs[0] + confs[1]) / 2
                                combinations.append((combined_two, avg_conf))
                        
                        # Thử clean và validate từng combination
                        for text_combo, combo_conf in combinations:
                            cleaned = clean_container_text(text_combo)
                            if cleaned and combo_conf > 0.2:  # Giảm threshold
                                result = ContainerResult(
                                    text=cleaned,
                                    confidence=combo_conf,
                                    method=f"PaddleOCR_v{variant_idx}"
                                )
                                all_results.append(result)
                                print(f"[SUCCESS] PaddleOCR variant {variant_idx}: {cleaned} (conf: {combo_conf:.3f})")
                        
                        if not combinations:
                            print(f"[INFO] PaddleOCR variant {variant_idx}: No valid combinations found")
                    else:
                        print(f"[INFO] PaddleOCR variant {variant_idx}: No valid text extracted")

                except Exception as e:
                    print(f"[ERROR] PaddleOCR variant {variant_idx} failed: {e}")

            # ===== EasyOCR =====
            if ocr_models.easy_ocr:
                try:
                    easy_results= ocr_models.easy_ocr.readtext(
                        processed_img,
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        width_ths=0.8,
                        height_ths=0.3,
                        paragraph=False,
                        slope_ths=0.1,
                        decoder="beamsearch",
                    )
                    if isinstance(easy_results, list) and easy_results:
                        # Lọc các kết quả hợp lệ (tuple/list có đủ >= 3 phần tử)
                        valid_easy = [r for r in easy_results if isinstance(r, (list, tuple)) and len(r) >= 3]
    
                        if valid_easy:
                            if orientation == "vertical":
                                valid_easy = sorted(valid_easy, key=lambda x: (x[0][0][1], x[0][0][0]))  # y then x
                            else:
                                valid_easy = sorted(valid_easy, key=lambda x: (x[0][0][1], x[0][0][0]))  # y then x
                            
                        if orientation == "vertical":
                            combined = ''.join([r[1] for r in valid_easy])
                            avg_conf = sum([r[2] for r in valid_easy]) / len(valid_easy)
                        else:
                            # Lấy kết quả có confidence cao nhất
                            best_result = max(valid_easy, key=lambda x: x[2])
                            combined = best_result[1]
                            avg_conf = best_result[2]
    
                        cleaned = clean_container_text(combined)
                        if cleaned and avg_conf > 0.3:
                            ocr_attempts.append((cleaned, avg_conf, f"EasyOCR_v{variant_idx}"))
                            print(f"[SUCCESS] EasyOCR container variant {variant_idx}: {cleaned} (conf: {avg_conf:.3f})")
                    else:
                        print(f"[INFO] EasyOCR container variant {variant_idx} returned no valid result")

                except Exception as e:
                    print(f"[ERROR] EasyOCR container variant {variant_idx} failed: {e}")


        # ===== Voting =====
        if len(all_results) >= 2:
            votes = {}
            for res in all_results:
                if res.text in votes:
                    votes[res.text]['count'] += 1
                    votes[res.text]['conf'] += res.confidence
                else:
                    votes[res.text] = {'count': 1, 'conf': res.confidence}
            for t, v in votes.items():
                if v['count'] >= 2:
                    avg_conf = v['conf'] / v['count']
                    bonus = 0.1 * (v['count'] - 1)
                    all_results.append(ContainerResult(
                        text=t,
                        confidence=min(avg_conf + bonus, 1.0),
                        method=f"Consensus_{v['count']}"
                    ))

        if all_results:
            def score(r):
                base = r.confidence
                method_bonus = 0.1 if 'PaddleOCR' in r.method else 0.05 if 'EasyOCR' in r.method else 0
                length_bonus = 0.05 if 8 <= len(r.text) <= 12 else -0.1
                return base + method_bonus + length_bonus

            all_results.sort(key=score, reverse=True)
            best = all_results[0]
            print(f"[SUCCESS] Best container result: {best.text} (conf: {best.confidence:.3f}, method: {best.method})")
            return best.text, best.confidence

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
            if conf < 0.4:
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
            result = extract_text_plate(cropped, model_manager.ocr_models)
            if result is None or not isinstance(result, tuple):
                print(f"[TIMEOUT] extract_text_plate timed out or failed for box: {[x1, y1, x2, y2]}")
                text, ocr_conf = None, 0.0
            else:
                text, ocr_conf = result
            
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

        results.append({
                    "type": "plate", 
                    "box": [x1, y1, x2, y2],
                    "text": None,
                    "confidence": conf,
                    "error": str(e)
                })
        
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

            result = extract_text_container(cropped, model_manager.ocr_models) # type: ignore
            if result is None or not isinstance(result, tuple):
                print(f"[TIMEOUT] extract_text_container timed out or failed for box: {[x1, y1, x2, y2]}")
                text, ocr_conf = None, 0.0
            else:
                text, ocr_conf = result

            results.append({
                "type": "container",
                "box": [x1, y1, x2, y2],
                "text": text,
                "confidence": ocr_conf if text else conf,
                "detection_confidence": conf,
                "ocr_confidence": ocr_conf if text else 0.0
            })
    
    except Exception as e:
        print(f"[FAILED] Enhanced container detection failed: {e}")
        results.append({
                    "type": "container", 
                    "box": [x1, y1, x2, y2],
                    "text": None,
                    "confidence": conf,
                    "error": str(e)
                })
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
            face_tensor = model_manager.face_transform(face_pil).unsqueeze(0).to(device) # type: ignore

            with torch.no_grad():
                embedding = model_manager.face_models['facenet'](face_tensor).cpu().numpy()

            # Classify face
            proba_list = model_manager.face_classifier.predict_proba(embedding)[0]
            best_idx = np.argmax(proba_list)
            best_prob = float(proba_list[best_idx])
            
            if best_prob >= FACE_CONFIDENCE_THRESHOLD:
                name = model_manager.label_encoder.inverse_transform([best_idx])[0]
            else:
                name = "Unknown"


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
        text = det["text"] if det["text"] else "None"

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

        print(f"[DEBUG] Processing frame shape: {frame.shape}")

        # Sequential detection to ensure OCR completes before moving on
        plate_detections = detect_plates(frame)
        detections.extend(plate_detections)
        print(f"[DEBUG] Plate detections: {plate_detections}")

        container_detections = detect_containers(frame)
        detections.extend(container_detections)
        print(f"[DEBUG] Container detections: {container_detections}")

        face_detections = detect_faces(frame)
        detections.extend(face_detections)

        metadata = extract_metadata(detections)
        print(f"[DEBUG] Final metadata: {metadata}")

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
        text = det.get("text", None)
        
        if text is not None and str(text).lower() != "none":
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
    """WebSocket endpoint được tối ưu cho xử lý batch images"""
    await websocket.accept()
    print("[WebSocket] Client connected")
    
    try:
        while True:
            try:
                # Không sử dụng timeout cho việc nhận tin nhắn
                # Thay vào đó dựa vào việc client gửi liên tục
                raw_message = await websocket.receive_text()
                
                message = json.loads(raw_message)
                
                # Xử lý batch images
                if message.get("type") == "batch":
                    images = message.get("images", [])
                    batch_results = []
                    
                    for idx, frame_b64 in enumerate(images):
                        if not frame_b64:
                            batch_results.append({
                                "image_index": idx,
                                "success": False,
                                "error": "No image data"
                            })
                            continue
                        
                        try:
                            # Decode và xử lý ảnh
                            img_data = base64.b64decode(frame_b64)
                            nparr = np.frombuffer(img_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is None:
                                batch_results.append({
                                    "image_index": idx,
                                    "success": False,
                                    "error": "Cannot decode image"
                                })
                                continue
                            
                            # Xử lý ảnh
                            start_time = time.time()
                            result = process_image(frame)
                            
                            if result is None:
                                batch_results.append({
                                    "image_index": idx,
                                    "success": False,
                                    "error": "Processing timeout"
                                })
                                continue
                            
                            annotated_frame, detections = result
                            processing_time = (time.time() - start_time) * 1000
                            
                            # Extract metadata
                            metadata = extract_metadata(detections)
                            
                            # Encode response (tùy chọn có trả về ảnh hay không)
                            annotated_b64 = None
                            if message.get("return_images", False):
                                _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                                annotated_b64 = base64.b64encode(buffer).decode('utf-8') # type: ignore
                            
                            batch_results.append({
                                "image_index": idx,
                                "success": True,
                                "image": annotated_b64,
                                "detections": detections,
                                "metadata": metadata,
                                "processing_time_ms": round(processing_time, 2)
                            })
                            
                            # Gửi kết quả ngay lập tức cho từng ảnh (streaming results)
                            if message.get("stream_results", False):
                                await websocket.send_text(json.dumps({
                                    "type": "progress",
                                    "image_index": idx,
                                    "total_images": len(images),
                                    "result": batch_results[-1]
                                }))
                        
                        except Exception as e:
                            print(f"[WebSocket Error] Processing image {idx}: {e}")
                            batch_results.append({
                                "image_index": idx,
                                "success": False,
                                "error": f"Processing error: {str(e)}"
                            })
                    
                    # Gửi kết quả cuối cùng cho toàn bộ batch
                    if not message.get("stream_results", False):
                        response = {
                            "type": "batch_complete",
                            "success": True,
                            "total_processed": len(images),
                            "results": batch_results
                        }
                        await websocket.send_text(json.dumps(response))
                
                # Xử lý single image (giữ nguyên logic cũ nhưng tối ưu)
                elif message.get("type") == "single" or "image" in message:
                    frame_b64 = message.get("image")
                    
                    if not frame_b64:
                        await websocket.send_text(json.dumps({
                            "success": False,
                            "error": "No image data received"
                        }))
                        continue
                    
                    # Decode và xử lý
                    img_data = base64.b64decode(frame_b64)
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        await websocket.send_text(json.dumps({
                            "success": False,
                            "error": "Cannot decode image"
                        }))
                        continue
                    
                    # Process image
                    start_time = time.time()
                    result = process_image(frame)
                    
                    if result is None:
                        await websocket.send_text(json.dumps({
                            "success": False,
                            "error": "Processing timeout"
                        }))
                        continue
                    
                    annotated_frame, detections = result
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Extract metadata và update global state
                    metadata = extract_metadata(detections)
                    update_latest_frame_and_metadata(annotated_frame, metadata)
                    
                    # Encode response
                    _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    annotated_b64 = base64.b64encode(buffer).decode('utf-8') # type: ignore
                    
                    response = {
                        "type": "single_complete",
                        "success": True,
                        "image": annotated_b64,
                        "detections": detections,
                        "metadata": metadata,
                        "processing_time_ms": round(processing_time, 2)
                    }
                    
                    await websocket.send_text(json.dumps(response))
                
                # Heartbeat/ping handling
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
                
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


# Thêm endpoint WebSocket tối ưu cho batch processing
@app.websocket("/ws/batch-detection")
async def websocket_batch_detection(websocket: WebSocket):
    """WebSocket endpoint chuyên biệt cho batch processing"""
    await websocket.accept()
    print("[WebSocket Batch] Client connected")
    
    try:
        while True:
            # Nhận message mà không timeout
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            
            images = message.get("images", [])
            if not images:
                await websocket.send_text(json.dumps({
                    "success": False,
                    "error": "No images provided"
                }))
                continue
            
            print(f"[Batch] Processing {len(images)} images")
            processed_count = 0
            
            for idx, frame_b64 in enumerate(images):
                try:
                    # Decode image
                    img_data = base64.b64decode(frame_b64)
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        continue
                    
                    # Process image
                    result = process_image(frame)
                    if result is None:
                        continue
                    
                    annotated_frame, detections = result
                    metadata = extract_metadata(detections)
                    processed_count += 1
                    
                    # Gửi progress update
                    progress_response = {
                        "type": "progress",
                        "image_index": idx,
                        "total_images": len(images),
                        "processed": processed_count,
                        "metadata": metadata,
                        "detections": len(detections)
                    }
                    
                    await websocket.send_text(json.dumps(progress_response))
                    
                    # Update global state with latest result
                    update_latest_frame_and_metadata(annotated_frame, metadata)
                    
                except Exception as e:
                    print(f"[Batch Error] Image {idx}: {e}")
                    continue
            
            # Send completion message
            completion_response = {
                "type": "batch_complete",
                "success": True,
                "total_images": len(images),
                "processed_count": processed_count,
                "completion_time": time.time()
            }
            
            await websocket.send_text(json.dumps(completion_response))
            
    except WebSocketDisconnect:
        print("[WebSocket Batch] Client disconnected")
    except Exception as e:
        print(f"[WebSocket Batch Error] {e}")


# Endpoint REST API thay thế cho batch processing
@app.post("/process-batch")
async def process_batch_images(request: Request):
    """REST API endpoint cho batch image processing"""
    request_id = str(uuid.uuid4())[:8]
    print(f"[{request_id}] Starting batch processing")

    try:
        data = await request.json() # type: ignore
        images = data.get("images", [])
        
        if not images:
            return JSONResponse({
                "success": False,
                "error": "No images provided"
            }, status_code=400)
        
        results = []
        for idx, frame_b64 in enumerate(images):
            try:
                # Decode image
                img_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    results.append({
                        "image_index": idx,
                        "success": False,
                        "error": "Cannot decode image"
                    })
                    print(f"[WARNING] Failed to decode image {idx}")
                    continue
                
                # Process image
                start_time = time.time()
                result = process_image(frame)
                
                if result is None:
                    results.append({
                        "image_index": idx,
                        "success": False,
                        "error": "Processing failed"
                    })
                    continue
                
                annotated_frame, detections = result
                processing_time = (time.time() - start_time) * 1000
                metadata = extract_metadata(detections)
                
                results.append({
                    "image_index": idx,
                    "success": True,
                    "detections": detections,
                    "metadata": metadata,
                    "processing_time_ms": round(processing_time, 2)
                })

                print(f"[DEBUG] Receiving batch with {len(images)} images")
                
            except Exception as e:
                results.append({
                    "image_index": idx,
                    "success": False,
                    "error": str(e)
                })
        print(f"[{request_id}] Results: {results}")
        return JSONResponse({
            "success": True,
            "total_processed": len([r for r in results if r["success"]]),
            "total_images": len(images),
            "results": results
        })
    
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# trường hợp timeout, timeout_with_executor(...) sẽ trả về None chứ không trả về Tuple (None, 0.0) gây lỗi NoneType

