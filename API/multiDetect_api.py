# multiDetect_api.py - FIXED VERSION
import asyncio
import subprocess
import time
import cv2
import torch
import json
import base64
import numpy as np
import joblib
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
latest_metadata = {"plate": "None", "container": "None", "face": "None"}
frame_lock = threading.Lock()

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== CONFIG =====
alphabet = sorted("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
device = "cuda" if torch.cuda.is_available() else "cpu"
char_conf = 0.9
MINDETECT = 0.95  # ngưỡng nhận diện face

# ===== LOAD MODEL =====
char_model = YOLO("modelAI/detect_Character.pt")
plate_model = YOLO("modelAI/detect_PlateNumber.pt")
container_model = YOLO("modelAI/detect_ContainerCode.pt")

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

# ===== Mapping bảng ký tự =====
char_to_index = {c: i + 1 for i, c in enumerate(alphabet)}
index_to_char = {i: c for c, i in char_to_index.items()}
index_to_char[0] = ""

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

# ===== Tách ký tự theo dòng =====
def group_char_to_1line(boxes, y_threshold=20):
    rows = []
    boxes = sorted(boxes, key=lambda b: b[1])
    for box in boxes:
        placed = False
        for row in rows:
            if abs(box[1] - row[0][1]) < y_threshold:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    for row in rows:
        row.sort(key=lambda b: b[0])
    rows.sort(key=lambda r: r[0][1])
    return rows

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

# ===== Detect Plates =====
def detect_plates(frame):
    results = []
    yolo_out = plate_model(frame)[0]
    for box in yolo_out.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Nếu confidence < 0.5 thì trả Unknown
        if conf < 0.5:
            results.append({
                "type": "plate",
                "box": [x1, y1, x2, y2],
                "text": None,
                "confidence": conf
            })
            continue

        # Crop vùng biển số và resize về size chuẩn (320×80)
        h, w = frame.shape[:2]
        x1_clamped = max(0, x1); y1_clamped = max(0, y1)
        x2_clamped = min(w, x2); y2_clamped = min(h, y2)
        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            continue
        cropped = cv2.resize(frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped], (320, 80))

        # Chạy detect ký tự trên crop
        char_out = char_model(cropped)[0]
        char_boxes = []
        for cbox in char_out.boxes:
            cconf = float(cbox.conf[0])
            cls_id = int(cbox.cls[0])
            pred_char = index_to_char.get(cls_id, "?") if cconf >= char_conf else "?"
            cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0])
            char_boxes.append([cx1, cy1, cx2, cy2, pred_char, cconf])

        grouped = group_char_to_1line(char_boxes)
        all_chars = [c for line in grouped for c in line]
        recognized = "".join([c[4] for c in all_chars])
        valid_chars = [c for c in all_chars if c[4] != "?"]
        acc = len(valid_chars) / len(all_chars) if all_chars else 0.0

        if acc >= 0.9:
            results.append({
                "type": "plate",
                "box": [x1, y1, x2, y2],
                "text": recognized,
                "confidence": acc
            })
        else:
            results.append({
                "type": "plate",
                "box": [x1, y1, x2, y2],
                "text": None,
                "confidence": acc
            })
    return results

# ===== Detect Container =====
def detect_containers(frame):
    results = []
    yolo_out = container_model(frame)[0]
    for box in yolo_out.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if conf < 0.5:
            results.append({
                "type": "container",
                "box": [x1, y1, x2, y2],
                "text": None,
                "confidence": conf
            })
            continue

        # Crop → detect ký tự
        h, w = frame.shape[:2]
        x1_clamped = max(0, x1); y1_clamped = max(0, y1)
        x2_clamped = min(w, x2); y2_clamped = min(h, y2)
        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            continue
        cropped = cv2.resize(frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped], (320, 80))

        char_out = char_model(cropped)[0]
        char_boxes = []
        for cbox in char_out.boxes:
            cconf = float(cbox.conf[0])
            cls_id = int(cbox.cls[0])
            pred_char = index_to_char.get(cls_id, "?") if cconf >= char_conf else "?"
            cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0])
            char_boxes.append([cx1, cy1, cx2, cy2, pred_char, cconf])

        grouped = group_char_to_1line(char_boxes)
        all_chars = [c for line in grouped for c in line]
        recognized = "".join([c[4] for c in all_chars])
        valid_chars = [c for c in all_chars if c[4] != "?"]
        acc = len(valid_chars) / len(all_chars) if all_chars else 0.0

        if acc >= 0.9:
            results.append({
                "type": "container",
                "box": [x1, y1, x2, y2],
                "text": recognized,
                "confidence": acc
            })
        else:
            results.append({
                "type": "container",
                "box": [x1, y1, x2, y2],
                "text": None,
                "confidence": acc
            })
    return results

# ===== Detect Faces =====
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
        "plate": (0, 255, 0),
        "container": (255, 0, 0),
        "face": (0, 0, 255)
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

# ===== Core xử lý 1 frame =====
def process_image(frame):
    detections = []
    detections += detect_plates(frame)
    detections += detect_containers(frame)
    detections += detect_faces(frame)
    annotated = draw_detections(frame.copy(), detections)
    return annotated, detections

# ===== Extract metadata từ detections =====
def extract_metadata(detections: List[Dict[str, Any]]) -> Dict[str, str]:
    """Trích xuất metadata từ detections cho Flutter"""
    metadata = {"plate": "None", "container": "None", "face": "None"}
    
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
    
    return metadata

# ===== Endpoint: Health check =====
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": device,
        "models": {
            "char_model": True,
            "plate_model": True,
            "container_model": True,
            "face_classifier": (classifier is not None),
            "label_encoder": (label_encoder is not None)
        }
    }

# ===== Endpoint: start-stream =====
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

# ===== Endpoint: video-feed =====
@app.get("/video-feed/combined-detection")
def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

# ===== WebSocket: combined detection - FIXED =====
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

# ===== WebSocket riêng cho Flutter metadata - NEW =====
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
