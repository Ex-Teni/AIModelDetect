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
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# ===== GLOBAL VARIABLES =====
app = FastAPI()
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

latest_frame = None
latest_metadata = {"plate": "None", "container": "None", "face": "None", "seal": "None"}
frame_lock = threading.Lock()

# ===== CORS MIDDLEWARE =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== CONFIGURATION =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# ===== MODEL INITIALIZATION =====
# YOLO models
plate_model = YOLO("modelAI/detect_PlateNumber.pt")
container_model = YOLO("modelAI/detect_ContainerCode.pt")

# TrOCR initialization
try:
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    print("TrOCR models loaded successfully")
except Exception as e:
    print(f"Error loading TrOCR: {e}")
    trocr_processor = None
    trocr_model = None

# Face recognition models
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
classifier = joblib.load('modelAI/face_classifier.joblib')
label_encoder = joblib.load('modelAI/label_encoder.joblib')

# Face preprocessing transform
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ===== UTILITY FUNCTIONS =====
def decode_base64_to_image(b64: str):
    """Decode base64 string to OpenCV image"""
    try:
        img_data = base64.b64decode(b64)
        arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def encode_image_to_base64(img) -> str:
    """Encode OpenCV image to base64 string"""
    _, buff = cv2.imencode(".jpg", img)
    return base64.b64encode(buff).decode("utf-8") # type: ignore

def timeout_handler_threaded(timeout_seconds):
    """Decorator for handling function timeouts using threading"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e # type: ignore
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                print(f"[TIMEOUT] {func.__name__} timed out after {timeout_seconds}s")
                return None, 0.0
            
            if exception[0]:
                print(f"[ERROR] {func.__name__} failed: {exception[0]}")
                return None, 0.0
                
            return result[0] if result[0] is not None else (None, 0.0)
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
    
    # Calculate overlap ratio
    overlap_ratio = intersection_area / min(box1_area, box2_area) if min(box1_area, box2_area) > 0 else 0
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
def clean_text(text, text_type):
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

# ===== OCR FUNCTIONS =====
@timeout_handler_threaded(OCR_TIMEOUT)
def extract_text_with_trocr_fast(image_crop, text_type="plate"):
    """Fast TrOCR extraction with timeout protection"""
    if trocr_processor is None or trocr_model is None:
        return None, 0.0
    
    try:
        # Simplified preprocessing
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop.copy()
        
        # Quick resize if too small
        h, w = gray.shape
        if h < 32 or w < 100:
            scale = max(32 / h, 100 / w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Quick TrOCR inference
        with torch.no_grad():
            inputs = trocr_processor(images=pil_image, return_tensors="pt").to(device) # type: ignore
            outputs = trocr_model.generate(**inputs, max_length=30)  # Reduced max_length for speed
            text = trocr_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip() # type: ignore

        if not text:
            return None, 0.0
        
        cleaned_text = clean_text(text, text_type)
        confidence = 0.8 if cleaned_text else 0.0
        return cleaned_text, confidence
        
    except Exception as e:
        print(f"[ERROR] Fast TrOCR failed: {e}")
        return None, 0.0

@timeout_handler_threaded(OCR_TIMEOUT)
def extract_text_with_easyocr_fast(image_crop, text_type="plate"):
    """Fast EasyOCR extraction as fallback"""
    try:
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
            
        results = reader.readtext(gray, detail=1, width_ths=0.7, height_ths=0.7)
        if not results:
            return None, 0.0
            
        best_result = max(results, key=lambda x: x[2]) # type: ignore
        text = best_result[1] # type: ignore
        confidence = best_result[2] # type: ignore
        cleaned_text = clean_text(text, text_type)
        return cleaned_text, confidence if cleaned_text else 0.0
        
    except Exception as e:
        print(f"[ERROR] Fast EasyOCR failed: {e}")
        return None, 0.0

# ===== DETECTION FUNCTIONS =====
def detect_plates(frame):
    """Detect license plates in frame"""
    results = []
    try:
        yolo_out = plate_model(frame, conf=0.1, iou=0.5)[0]
        
        for box in yolo_out.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Add low confidence detections without OCR
            if conf < 0.1:
                results.append({
                    "type": "plate",
                    "box": [x1, y1, x2, y2],
                    "text": None,
                    "confidence": conf
                })
                continue

            # Crop with minimal padding
            h, w = frame.shape[:2]
            padding = 5
            x1_c = max(0, x1 - padding)
            y1_c = max(0, y1 - padding)
            x2_c = min(w, x2 + padding)
            y2_c = min(h, y2 + padding)
            
            if x2_c <= x1_c or y2_c <= y1_c:
                continue
                
            cropped = frame[y1_c:y2_c, x1_c:x2_c]

            # Try TrOCR first, then EasyOCR as fallback
            text, ocr_conf = extract_text_with_trocr_fast(cropped, "plate")
            if not text:
                text, ocr_conf = extract_text_with_easyocr_fast(cropped, "plate")
            
            results.append({
                "type": "plate",
                "box": [x1, y1, x2, y2],
                "text": text,
                "confidence": ocr_conf if text else conf
            })
    
    except Exception as e:
        print(f"[ERROR] Plate detection failed: {e}")
    
    return results

def detect_containers(frame):
    """Detect container codes in frame"""
    results = []
    try:
        yolo_out = container_model(frame, conf=0.15, iou=0.5)[0]
        
        for box in yolo_out.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Add low confidence detections without OCR
            if conf < 0.15:
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
            text, ocr_conf = extract_text_with_trocr_fast(cropped, "container")
            
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
    if classifier is None or label_encoder is None:
        return []

    results = []
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                embedding = facenet(face_tensor).cpu().numpy()

            # Classify face
            proba_list = classifier.predict_proba(embedding)[0]
            best_idx = np.argmax(proba_list)
            best_prob = float(proba_list[best_idx])
            
            name = label_encoder.inverse_transform([best_idx])[0] if best_prob >= FACE_CONFIDENCE_THRESHOLD else None

            results.append({
                "type": "face",
                "box": [x1c, y1c, x2c, y2c],
                "text": name,
                "confidence": best_prob
            })

        return results

    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
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
@timeout_handler_threaded(IMAGE_PROCESSING_TIMEOUT)
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
    global latest_frame, latest_metadata
    with frame_lock:
        latest_frame = frame.copy()
        latest_metadata = metadata.copy()

def generate_mjpeg():
    """Generate MJPEG stream for video feed"""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                current_frame = latest_frame.copy()
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

# ===== API ENDPOINTS =====
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": device,
        "models": {
            "easyocr_reader": True,
            "plate_model": True,
            "container_model": True,
            "trocr_model": (trocr_processor is not None and trocr_model is not None),
            "face_classifier": (classifier is not None),
            "label_encoder": (label_encoder is not None),
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
            with frame_lock:
                current_metadata = latest_metadata.copy()
            
            # Only send if metadata has changed
            if current_metadata != last_metadata:
                await websocket.send_json(current_metadata)
                last_metadata = current_metadata.copy()
            
            await asyncio.sleep(0.1)  # 100ms polling interval
            
    except WebSocketDisconnect:
        print("[Flutter WebSocket] Client disconnected")
    except Exception as e:
        print(f"[Flutter WebSocket Error] {e}")

# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        ws_ping_interval=PING_INTERVAL,
        ws_ping_timeout=PING_TIMEOUT
    )

