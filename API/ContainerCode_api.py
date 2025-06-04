import base64
import json
import threading
import time
import cv2
from fastapi.responses import StreamingResponse
import numpy as np
from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import torch
import subprocess

app = FastAPI()

# ========== Config ==========
alphabet = sorted("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
device = "cuda" if torch.cuda.is_available() else "cpu"
char_conf = 0.9  # Chỉ lấy ký tự nếu confidence > 90%

# ========== Model ==========
char_model = YOLO("modelAI/detect_Character.pt")
container_model = YOLO("modelAI/detect_ContainerCode.pt")

# ========== Mapping ==========
char_to_index = {c: i + 1 for i, c in enumerate(alphabet)}
index_to_char = {i: c for c, i in char_to_index.items()}
index_to_char[0] = ""

# ========== Shared Frame ==========
latest_frame = None
frame_lock = threading.Lock()


# ========== Helper Functions ==========
def decode_base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8") # type: ignore


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


# ========== Streaming Endpoint ==========
@app.get("/video-feed/container-detection")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    _, jpeg = cv2.imencode(".jpg", latest_frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # 30 fps
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/start-stream")
def start_stream():
    subprocess.Popen(["python", "client_camera.py"])
    return {"status": "streaming started"}


# ========== WebSocket Endpoint ==========
@app.websocket("/ws/container-detection")
async def websocket_container_detection(websocket: WebSocket):
    await websocket.accept()
    global latest_frame
    print(f"[INFO] ContainerCode API connected: {websocket.client}")

    while True:
        try:
            data = await websocket.receive_text()
            try:
                incoming_data = json.loads(data)
            except Exception as ex:
                print("[DEBUG] json.loads failed:", ex)
                continue

            if "image" not in incoming_data:
                print("[DEBUG] No 'image' key in incoming_data:", incoming_data.keys())
                continue

            frame = decode_base64_to_image(incoming_data["image"])
            detect_result = container_model(frame)[0]

            container_results = []

            for container_box in detect_result.boxes:
                container_conf = float(container_box.conf[0])
                x1, y1, x2, y2 = map(int, container_box.xyxy[0])
                cropped_container = cv2.resize(frame[y1:y2, x1:x2], (320, 80))

                if container_conf < 0.8:
                    label_text = "[CONTAINER]_Unknown"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label_text, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    container_results.append({
                        "box": [x1, y1, x2, y2],
                        "plate": "[CONTAINER]_Unknown"
                    })
                    continue

                char_result = char_model(cropped_container)[0]

                char_result = char_model(cropped_container)[0]
                character_boxes = []
                for char_box in char_result.boxes:
                    confidence = float(char_box.conf[0])
                    class_id = int(char_box.cls[0])
                    predicted_char = index_to_char.get(class_id, "?") if confidence >= char_conf else "?"

                    cx1, cy1, cx2, cy2 = map(int, char_box.xyxy[0])
                    character_boxes.append([cx1, cy1, cx2, cy2, predicted_char, confidence])

                grouped_lines = group_char_to_1line(character_boxes)
                all_chars = [c for line in grouped_lines for c in line]
                recognized_text = ''.join(c[4] for c in all_chars)

                valid_chars = [c for c in all_chars if c[4] != "?"]
                accuracy = len(valid_chars) / len(all_chars) if all_chars else 0.0

                if accuracy >= 0.9:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{recognized_text} ({accuracy*100:.1f}%)"
                    cv2.putText(frame, label_text, (x1, max(y1 - 10, 0)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),    2)                                

                    container_results.append({
                        "box": [x1, y1, x2, y2],
                        "code": recognized_text
                    })

            # Cập nhật frame
            with frame_lock:
                latest_frame = frame.copy()

            # Gửi kết quả qua WebSocket
            encoded_frame = encode_image_to_base64(frame)
            await websocket.send_text(json.dumps({
                "containers": container_results,
                "image_base64": encoded_frame
            }))

        except Exception as e:
            print("[ERROR] WebSocket Error:", e)
            break
