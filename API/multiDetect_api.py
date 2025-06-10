import asyncio
import subprocess
import time
import cv2
import torch
import json
import base64
import numpy as np
import joblib
import easyocr
import re
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from starlette.websockets import WebSocketDisconnect
import threading
from typing import List, Dict, Any

app = FastAPI()

# ====== Cải tiến OCR =====
def preprocess_image_for_ocr(image):
    """
    Tiền xử lý ảnh để cải thiện độ chính xác OCR
    """
    # Chuyển sang grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Làm mờ nhẹ để giảm noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Threshold để tạo ảnh đen trắng rõ nét
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def detect_rotation_angle(image):

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tính góc nghiêng
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Điều chỉnh góc
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    return angle

def rotate_image(image, angle):
    
    if abs(angle) < 5:  # Không cần xoay nếu góc quá nhỏ
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Ma trận xoay
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Tính kích thước mới
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_angle) + (w * cos_angle))
    new_h = int((h * cos_angle) + (w * sin_angle))
    
    # Điều chỉnh ma trận để không bị cắt
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Xoay ảnh
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def extract_plate_text_advanced(image_crop):
    try:
        # Tiền xử lý ảnh
        processed = preprocess_image_for_ocr(image_crop)
        
        # Resize để cải thiện OCR
        height, width = processed.shape
        if height < 60:
            scale = 60 / height
            new_width = int(width * scale)
            processed = cv2.resize(processed, (new_width, 60), interpolation=cv2.INTER_CUBIC)
        
        # Chuyển sang RGB cho EasyOCR
        if len(processed.shape) == 2:
            rgb_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Sử dụng EasyOCR với cấu hình tối ưu cho biển số
        results = easyocr_reader.readtext(
            rgb_image,
            detail=1,
            paragraph=False,  # Đọc từng dòng riêng biệt
            width_ths=0.2,   # Ngưỡng để tách các từ
            height_ths=0.2   # Ngưỡng để tách các dòng
        )
        
        if not results:
            return None, 0.0
        
        # Xử lý kết quả cho biển số 2 dòng
        texts = []
        confidences = []
        
        # Sắp xếp results theo tọa độ Y (từ trên xuống dưới)
        results_sorted = sorted(results, key=lambda x: x[0][0][1])  # type: ignore 
        
        for result in results_sorted:
            bbox, text, confidence = result
            if confidence >= OCR_CONFIDENCE_THRESHOLD: # type: ignore
                # Làm sạch text
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(cleaned_text) >= 2:  # Chỉ giữ text có ít nhất 2 ký tự
                    texts.append(cleaned_text)
                    confidences.append(confidence)
        
        if not texts:
            return None, 0.0
        
        # Ghép các dòng text lại (với dấu cách hoặc dấu gạch ngang)
        if len(texts) > 1:
            final_text = '-'.join(texts)  # Hoặc ' '.join(texts)
        else:
            final_text = texts[0]
        
        avg_confidence = sum(confidences) / len(confidences)
        
        return final_text, avg_confidence
        
    except Exception as e:
        print(f"[ERROR] Advanced plate OCR failed: {e}")
        return None, 0.0

def extract_container_text_advanced(image_crop):
    try:
        # Tiền xử lý ảnh
        processed = preprocess_image_for_ocr(image_crop)
        
        # Phát hiện góc xoay
        angle = detect_rotation_angle(processed)
        
        # Thử nhiều góc xoay khác nhau
        angles_to_try = [0, angle, 90, -90, 180]
        best_result = None
        best_confidence = 0.0
        
        for test_angle in angles_to_try:
            # Xoay ảnh
            if test_angle != 0:
                rotated = rotate_image(processed, test_angle)
            else:
                rotated = processed
            
            # Resize để cải thiện OCR
            height, width = rotated.shape
            if height < 80 or width < 200:
                scale = max(80 / height, 200 / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                rotated = cv2.resize(rotated, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Chuyển sang RGB
            if len(rotated.shape) == 2:
                rgb_image = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            
            # OCR với EasyOCR
            results = easyocr_reader.readtext(
                rgb_image,
                detail=1,
                paragraph=True,   # Ghép thành đoạn văn
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # Chỉ cho phép chữ và số
            )
            
            for result in results:
                bbox, text, confidence = result
                if confidence > best_confidence: # type: ignore
                    # Làm sạch text container
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(cleaned_text) >= 4:  # Container code thường dài ít nhất 4 ký tự
                        best_result = cleaned_text
                        best_confidence = confidence
        
        if best_result and best_confidence >= OCR_CONFIDENCE_THRESHOLD: # type: ignore
            return best_result, best_confidence
        else:
            return None, best_confidence
            
    except Exception as e:
        print(f"[ERROR] Advanced container OCR failed: {e}")
        return None, 0.0

# ===== GLOBAL VARIABLES =====
latest_frame = None
latest_metadata = {"plate": "None", "container": "None", "face": "None", "seal": "None"}
frame_lock = threading.Lock()

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== CONFIG =====
device = "cuda" if torch.cuda.is_available() else "cpu"
MINDETECT = 0.95  # ngưỡng nhận diện face
OCR_CONFIDENCE_THRESHOLD = 0.5  # ngưỡng tin cậy cho EasyOCR
SEAL_OCR_CONFIDENCE_THRESHOLD = 0.3  # ngưỡng thấp hơn cho Seal vì có thể khó đọc

# ===== LOAD MODEL =====
plate_model = YOLO("modelAI/detect_PlateNumber.pt")
container_model = YOLO("modelAI/detect_ContainerCode.pt")

easyocr_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))

mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

try:
    classifier = joblib.load('modelAI/face_classifier.joblib')
    label_encoder = joblib.load('modelAI/label_encoder.joblib')
except Exception:
    print("[WARNING] Face recognition models not found")
    classifier = None
    label_encoder = None

# ===== Transform face =====
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ===== Helper: decode/encode base64 ↔ image =====
''' === THÊM CẤU HÌNH CHO EASYOCR === '''
def initialize_easyocr_reader():
    """
    Khởi tạo EasyOCR reader với cấu hình tối ưu
    """
    return easyocr.Reader(
        ['en'], 
        gpu=(device == 'cuda'),
        model_storage_directory='easyocr_models',  # Thư mục lưu models
        download_enabled=True,
        detector=True,
        recognizer=True,
        verbose=False
    )

def decode_base64_to_image(b64: str):
    try:
        img_data = base64.b64decode(b64)
        arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def encode_image_to_base64(img) -> str:
    _, buff = cv2.imencode(".jpg", img)
    return base64.b64encode(buff).decode("utf-8") # type: ignore

# ===== EasyOCR function to replace character detection =====
def extract_text_with_easyocr(image_crop, confidence_threshold=OCR_CONFIDENCE_THRESHOLD):
    try:
        # Chuyển đổi sang RGB nếu cần
        if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_crop
        
        # Sử dụng EasyOCR để đọc text
        results = easyocr_reader.readtext(image_rgb)
        
        if not results:
            return None, 0.0
        
        # Lấy kết quả có confidence cao nhất
        best_result = max(results, key=lambda x: x[2]) # type: ignore
        text = best_result[1].strip() # type: ignore
        confidence = float(best_result[2]) # type: ignore
        
        # Chỉ trả về kết quả nếu confidence >= threshold
        if confidence >= confidence_threshold:
            return text, confidence
        else:
            return None, confidence
            
    except Exception as e:
        print(f"[ERROR] EasyOCR extraction failed: {e}")
        return None, 0.0

# ===== Direct Seal OCR Detection =====
def detect_seal_text(frame):
    results = []
    try:
        # Chuyển đổi sang RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = frame
        
        # Sử dụng EasyOCR để đọc tất cả text trong frame
        ocr_results = easyocr_reader.readtext(image_rgb)
        
        # Lấy các bounding box từ plate và container models để loại trừ
        plate_boxes = []
        container_boxes = []
        
        # Detect plates để loại trừ
        plate_detections = plate_model(frame)[0]
        for box in plate_detections.boxes:
            if float(box.conf[0]) > 0.1:  # Chỉ lấy các detection có confidence > 0.1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_boxes.append([x1, y1, x2, y2])
        
        # Detect containers để loại trừ
        container_detections = container_model(frame)[0]
        for box in container_detections.boxes:
            if float(box.conf[0]) > 0.1:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                container_boxes.append([x1, y1, x2, y2])
        
        # Xử lý từng text detection từ EasyOCR
        for detection in ocr_results:
            bbox, text, confidence = detection
            
            # Chỉ xử lý nếu confidence >= threshold
            if confidence < SEAL_OCR_CONFIDENCE_THRESHOLD: # type: ignore
                continue
                
            # Chuyển đổi bbox từ EasyOCR format sang [x1, y1, x2, y2]
            points = np.array(bbox, dtype=int)
            x1 = int(np.min(points[:, 0]))
            y1 = int(np.min(points[:, 1]))
            x2 = int(np.max(points[:, 0]))
            y2 = int(np.max(points[:, 1]))
            
            # Kiểm tra xem text này có nằm trong vùng plate hoặc container không
            is_in_existing_detection = False
            
            # Kiểm tra overlap với plate boxes
            for plate_box in plate_boxes:
                if boxes_overlap([x1, y1, x2, y2], plate_box):
                    is_in_existing_detection = True
                    break
            
            # Kiểm tra overlap với container boxes
            if not is_in_existing_detection:
                for container_box in container_boxes:
                    if boxes_overlap([x1, y1, x2, y2], container_box):
                        is_in_existing_detection = True
                        break
            
            # Nếu không overlap với plate/container và có text hợp lệ
            if not is_in_existing_detection and text.strip():
                # Filter cho Seal - có thể thêm logic filter ở đây
                filtered_text = filter_seal_text(text.strip())
                if filtered_text:
                    results.append({
                        "type": "seal",
                        "box": [x1, y1, x2, y2],
                        "text": filtered_text,
                        "confidence": confidence
                    })
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Seal text detection failed: {e}")
        return []

def boxes_overlap(box1, box2, threshold=0.3):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Tính intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Tính tỷ lệ overlap
    overlap_ratio = intersection_area / min(box1_area, box2_area)
    return overlap_ratio > threshold

def filter_seal_text(text):
    # Loại bỏ text quá ngắn
    if len(text) < 3:
        return None
    
    # Loại bỏ text chỉ chứa số (có thể là từ plate)
    if text.isdigit():
        return None
    
    # Loại bỏ text chỉ chứa ký tự đặc biệt
    if not any(c.isalnum() for c in text):
        return None
    
    return text.upper().strip()

# ===== Streamming =====
def update_latest_frame_and_metadata(frame, metadata):
    global latest_frame, latest_metadata
    with frame_lock:
        latest_frame = frame.copy()
        latest_metadata = metadata.copy()

def generate_mjpeg():
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

# ===== Detect Plates - MODIFIED TO USE EASYOCR =====
def detect_plates(frame):
    results = []
    yolo_out = plate_model(frame)[0]
    
    for box in yolo_out.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if conf < 0.1:
            results.append({
                "type": "plate",
                "box": [x1, y1, x2, y2],
                "text": None,
                "confidence": conf
            })
            continue

        # Crop vùng biển số
        h, w = frame.shape[:2]
        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(w, x2)
        y2_clamped = min(h, y2)
        
        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            continue
            
        cropped = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

        # SỬ DỤNG HÀM OCR NÂNG CAO
        recognized_text, ocr_confidence = extract_plate_text_advanced(cropped)
        
        results.append({
            "type": "plate",
            "box": [x1, y1, x2, y2],
            "text": recognized_text,
            "confidence": ocr_confidence if recognized_text else conf
        })
    
    return results


# ===== Detect Container - MODIFIED TO USE EASYOCR =====
def detect_containers(frame):
    results = []
    yolo_out = container_model(frame)[0]
    
    for box in yolo_out.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if conf < 0.1:
            results.append({
                "type": "container",
                "box": [x1, y1, x2, y2],
                "text": None,
                "confidence": conf
            })
            continue

        # Crop vùng container
        h, w = frame.shape[:2]
        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(w, x2)
        y2_clamped = min(h, y2)
        
        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            continue
            
        cropped = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

        # SỬ DỤNG HÀM OCR NÂNG CAO
        recognized_text, ocr_confidence = extract_container_text_advanced(cropped)
        
        results.append({
            "type": "container",
            "box": [x1, y1, x2, y2],
            "text": recognized_text,
            "confidence": ocr_confidence if recognized_text else conf
        })
    
    return results

# ===== Detect Faces - UNCHANGED =====
def detect_faces(frame):
    if classifier is None or label_encoder is None:
        return []

    results = []
    try:
        # MTCNN detect trả về bounding boxes + probabilities
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(img_rgb) # type: ignore

        # Nếu không có boxes hoặc tất cả prob < MINDETECT, trả về [] (không thấy face)
        if boxes is None or len(boxes) == 0:
            return []

        for i, box in enumerate(boxes):
            prob = float(probs[i]) if probs is not None else 0.0
            if prob < MINDETECT:
                # Xem như không detect đủ tin cậy → skip
                continue

            x1, y1, x2, y2 = map(int, box)
            # Chắc chắn crop không vượt ngoài ảnh
            h, w = img_rgb.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                continue

            face_crop = img_rgb[y1c:y2c, x1c:x2c]
            face_pil = Image.fromarray(face_crop)
            face_tensor = face_transform(face_pil).unsqueeze(0).to(device) # type: ignore

            with torch.no_grad():
                embedding = facenet(face_tensor).cpu().numpy()

            # Dự đoán tên với classifier đã load
            proba_list = classifier.predict_proba(embedding)[0]
            best_idx = np.argmax(proba_list)
            best_prob = float(proba_list[best_idx])
            if best_prob >= MINDETECT:
                name = label_encoder.inverse_transform([best_idx])[0]
            else:
                name = None  # Không đủ tin cậy thì trả None

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

# ===== Vẽ bounding box lên frame =====
def draw_detections(frame, detections):
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det_type}: {text} ({conf:.2f})"
        cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# ===== Core xử lý 1 frame - UPDATED =====
def process_image(frame):
    detections = []
    detections += detect_plates(frame)
    detections += detect_containers(frame)
    detections += detect_faces(frame)
    detections += detect_seal_text(frame)  # NEW: Add seal detection
    annotated = draw_detections(frame.copy(), detections)
    return annotated, detections

# ===== Extract metadata từ detections - UPDATED =====
def extract_metadata(detections: List[Dict[str, Any]]) -> Dict[str, str]:
    metadata = {"plate": "None", "container": "None", "face": "None", "seal": "None"}
    
    for det in detections:
        det_type = det["type"]
        text = det["text"]
        
        if text is not None and text != "None":
            if det_type == "plate":
                metadata["plate"] = text
            elif det_type == "container":
                metadata["container"] = text
            elif det_type == "face":
                metadata["face"] = text
            elif det_type == "seal":  # NEW
                metadata["seal"] = text
    
    return metadata

# ===== Endpoint: Health check - UPDATED =====
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": device,
        "models": {
            "easyocr_reader": True,
            "plate_model": True,
            "container_model": True,
            "face_classifier": (classifier is not None),
            "label_encoder": (label_encoder is not None),
            "seal_detection": True  # NEW
        }
    }

# ===== Rest of the endpoints remain unchanged =====
@app.get("/start-stream")
def start_stream():
    try:
        # Chạy client_camera.py trong background
        subprocess.Popen(["python", "client_camera.py"])
        return JSONResponse({"message": "Client camera started"}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": f"Cannot start client_camera: {e}"}, status_code=500)

@app.get("/video-feed/combined-detection")
def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/combined-detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected")
    
    try:
        while True:
            # Nhận JSON từ client_camera
            raw_message = await websocket.receive_text()
            
            try:
                # Parse JSON message
                message = json.loads(raw_message)
                frame_b64 = message.get("image")
                
                if not frame_b64:
                    await websocket.send_text(json.dumps({
                        "success": False,
                        "error": "No image data received"
                    }))
                    continue
                
                # Decode frame từ base64
                img_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_text(json.dumps({
                        "success": False,
                        "error": "Cannot decode image"
                    }))
                    continue
                
                # GỌI THỰC TẾ các AI models
                annotated_frame, detections = process_image(frame)
                
                # Trích xuất metadata cho Flutter
                metadata = extract_metadata(detections)
                
                # Cập nhật global variables cho MJPEG stream (thread-safe)
                update_latest_frame_and_metadata(annotated_frame, metadata)
                
                # Encode annotated frame về base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                annotated_b64 = base64.b64encode(buffer).decode('utf-8') # type: ignore
                
                # Trả response đúng format cho client_camera
                response = {
                    "success": True,
                    "image": annotated_b64,
                    "detections": detections,
                    "metadata": metadata
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
async def flutter_metadata_endpoint(websocket: WebSocket):
    """WebSocket riêng để gửi metadata cho Flutter app"""
    await websocket.accept()
    print("[Flutter WebSocket] Client connected")
    
    try:
        last_metadata = None
        while True:
            # Lấy metadata hiện tại (thread-safe)
            with frame_lock:
                current_metadata = latest_metadata.copy()
            
            # Chỉ gửi khi có thay đổi
            if current_metadata != last_metadata:
                await websocket.send_json(current_metadata)
                last_metadata = current_metadata.copy()
            
            # Sleep ngắn để không spam
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print("[Flutter WebSocket] Client disconnected")
    except Exception as e:
        print(f"[Flutter WebSocket Error] {e}")
