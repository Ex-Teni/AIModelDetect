# multiDetect_api.py - MODIFIED WITH EASYOCR + SEAL DETECTION
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

# EasyOCR Reader - khởi tạo một lần
easyocr_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))

# MTCNN + FaceNet
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load classifier và label_encoder (dùng joblib)
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
    """
    Sử dụng EasyOCR để nhận diện text từ crop image
    Trả về text và confidence score
    """
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

# ===== NEW: Direct Seal OCR Detection =====
def detect_seal_text(frame):
    """
    Detect Seal text directly from frame using EasyOCR
    This function scans the entire frame for text that doesn't belong to plates or containers
    """
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
    """
    Kiểm tra xem 2 bounding box có overlap không
    threshold: tỷ lệ overlap tối thiểu để coi là overlap
    """
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
    """
    Filter text để chỉ giữ lại những text có thể là Seal
    Có thể customize logic này theo yêu cầu cụ thể
    """
    # Loại bỏ text quá ngắn
    if len(text) < 3:
        return None
    
    # Loại bỏ text chỉ chứa số (có thể là từ plate)
    if text.isdigit():
        return None
    
    # Loại bỏ text chỉ chứa ký tự đặc biệt
    if not any(c.isalnum() for c in text):
        return None
    
    # Có thể thêm thêm logic filter khác ở đây
    # Ví dụ: filter theo pattern của Seal number
    
    return text.upper().strip()

# ===== Streamming =====
def update_latest_frame_and_metadata(frame, metadata):
    """Thread-safe update của frame và metadata"""
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
        
        # Nếu confidence < 0.1 thì trả Unknown
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
        
        # Resize để cải thiện OCR (tùy chọn)
        if cropped.shape[0] < 50 or cropped.shape[1] < 150:
            scale_factor = max(50 / cropped.shape[0], 150 / cropped.shape[1])
            new_height = int(cropped.shape[0] * scale_factor)
            new_width = int(cropped.shape[1] * scale_factor)
            cropped = cv2.resize(cropped, (new_width, new_height))

        # Sử dụng EasyOCR thay vì model AI
        recognized_text, ocr_confidence = extract_text_with_easyocr(cropped)
        
        if recognized_text is not None:
            results.append({
                "type": "plate",
                "box": [x1, y1, x2, y2],
                "text": recognized_text,
                "confidence": ocr_confidence
            })
        else:
            results.append({
                "type": "plate",
                "box": [x1, y1, x2, y2],
                "text": None,
                "confidence": ocr_confidence
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
        
        # Resize để cải thiện OCR (tùy chọn)
        if cropped.shape[0] < 50 or cropped.shape[1] < 150:
            scale_factor = max(50 / cropped.shape[0], 150 / cropped.shape[1])
            new_height = int(cropped.shape[0] * scale_factor)
            new_width = int(cropped.shape[1] * scale_factor)
            cropped = cv2.resize(cropped, (new_width, new_height))

        # Sử dụng EasyOCR thay vì model AI
        recognized_text, ocr_confidence = extract_text_with_easyocr(cropped)
        
        if recognized_text is not None:
            results.append({
                "type": "container",
                "box": [x1, y1, x2, y2],
                "text": recognized_text,
                "confidence": ocr_confidence
            })
        else:
            results.append({
                "type": "container",
                "box": [x1, y1, x2, y2],
                "text": None,
                "confidence": ocr_confidence
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
    """Trích xuất metadata từ detections cho Flutter"""
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
    """
    Khi Flutter bấm 'Stream', FastAPI sẽ khởi chạy client_camera.py như subprocess.
    Cần chắc rằng client_camera.py nằm cùng thư mục hoặc có đường dẫn chính xác.
    """
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
