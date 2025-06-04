import base64
import json
import threading
import time
import cv2
from fastapi.responses import StreamingResponse
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket
from pydantic import BaseModel
from ultralytics import YOLO
import torch 
from typing import List 
import subprocess

import websocket


# === Config ===
alphabet = sorted("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
device = "cuda" if torch.cuda.is_available() else "cpu"
char_conf = 0.9 # Chỉ detect ký tự nào có xác xuất >90%

# === Model AI ====
char_model = YOLO("modelAI/detect_Character.pt") # model chữ
container_model = YOLO("modelAI/") # model số seal

# === Mapping Char ===
char_to_index = {c: i +1 for i, c in enumerate(alphabet)}
index_to_char = {i: c for c, i in char_to_index.items()}
index_to_char[0] = ""

# === Streaming frame setup ===
latest_frame = None
frame_lock = threading.Lock()
app = FastAPI()

# === Group char ===
def group_char_to_1line(boxes, y_threshold=20): # Ghép 2 dòng lại thành 1
    rows = []
    boxes = sorted(boxes, key = lambda b: b[1])
    for box in boxes:
        placed = True
        for row in rows:
            if abs(box[1] - row[0][1]) < y_threshold:
                row.append(box)
                placed = True
                break
        if not placed: 
            rows.append([box])
    for row in rows:
        row.sort(key = lambda b: b[0])
    rows.sort(key = lambda r: r[0][1])
    return rows

def decode_base64_to_image(base64_str): # Nhận dữ liệu base64 từ client_camera decode lại thành ảnh
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image_to_base64(image): # Encode ảnh lại thành base64 gửi dữ liệu đến frontend
    _, buffer = cv2.imencode(".jpg" ,image)
    return base64.b64encode(buffer).decode("utf-8") # type: ignore


#==================== Stream Video ======================
@app.get("/video-feed/seal-number")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    _, jpeg = cv2.imencode(".jpg", latest_frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03) #30fps
    return StreamingResponse(generate(), media_type = "multipart/x-mixed-replace; boundary=frame")

# ======================== API ==========================

# === WebSocket ===
@app.websocket("/ws/seal-detection")
async def websocket_container_code(websocket: WebSocket):
    await websocket.accept()
    global latest_frame

    while True:
        try:
            data = await websocket.receive_text() # type: ignore
            incoming_data = json.loads(data)

            frame = decode_base64_to_image(incoming_data["image"])
            detect_result = container_model(frame)[0]

            container_results = []

            for container_box in detect_result.boxes:
                container_conf = float(container_box.confidence[0])
                if container_conf < 0.9:
                    continue  # bỏ qua nếu container box < 90%

                x1, y1, x2, y2 = map(int, container_box.xyxy[0])
                plate_crop = cv2.resize(frame[y1:y2, x1:x2], (320, 80))
                char_result = char_model(plate_crop)[0]

                character_boxes = []
                for char_box in char_result.boxes:
                    confidence = float(char_box.conf[0])
                    class_id = int(char_box.cls[0])
                    predicted_char = index_to_char.get(class_id, "?") if confidence >= char_conf else "?"
                    
                    cx1, cy1, cx2, cy2 = map(int, char_box.xyxy[0])
                    character_boxes.append([cx1, cy1, cx2, cy2, predicted_char, confidence])

                grouped_character_lines = group_char_to_1line,(character_boxes)
                all_chars = [c for line in grouped_character_lines for c in line] # type: ignore
                recognized_plate_text = ''.join(c[4] for c in all_chars)

                valid_chars = [c for c in all_chars if c[4] != "?"]
                accuracy = len(valid_chars) / len(all_chars) if all_chars else 0.0


                # Vẽ bounding box và text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, recognized_plate_text, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                container_results.append({
                    "box": [x1, y1, x2, y2],
                    "code": recognized_plate_text
                })

            # Cập nhật frame mới nhất để stream
            with frame_lock:
                global latest_frame
                latest_frame = frame.copy()
                print("Updated latest frame")

            encoded_frame = encode_image_to_base64(frame)

            await websocket.send_text(json.dumps({
                "seals": container_results,
                "image_base64": encoded_frame
            }))

        except Exception as e:
            print("[ERROR] WebSocket Error:", e)
            break
