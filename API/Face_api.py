
import torch
import base64
import json
import threading
import time
import cv2
import numpy as np
from mtcnn import MTCNN
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
from PIL import Image
from torch import device
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from joblib import load
from torchvision import transforms

# Load model
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device) # type: ignore
classifier = load('modelAI/face_classifier.joblib')
label_encoder = load('modelAI/label_encoder.joblib')

MINDETECT = 0.95

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def perform_recognize_face(frame_np: np.ndarray):
    try:
        # Chuyển BGR -> RGB (OpenCV dùng BGR, PIL dùng RGB)
        img_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)
        img_rgb = np.array(image)
    except Exception as e:
        print("Failed to convert image:", e)
        return []

    results = []
    boxes, _ = mtcnn.detect(img_rgb)  # type: ignore

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_img = img_rgb[y1:y2, x1:x2]
            try:
                face_pil = Image.fromarray(face_img)
                face_tensor = transform(face_pil).unsqueeze(0).to(device)  # type: ignore

                with torch.no_grad():
                    embedding = facenet(face_tensor).cpu().numpy()

                proba_list = classifier.predict_proba(embedding)[0]
                best_pred_idx = np.argmax(proba_list)
                best_proba = proba_list[best_pred_idx]

                name = label_encoder.inverse_transform([best_pred_idx])[0] if best_proba >= MINDETECT else "Unknown"

                results.append({
                    "name": name,
                    "box": [x1, y1, x2, y2],
                    "probability": round(float(best_proba), 2)
                })
            except Exception as e:
                print("Error recognizing face:", e)
                continue

    return results

# === FastAPI App ===
app = FastAPI()

# === Config ===
latest_frame = None
frame_lock = threading.Lock()

# === Helper ===

def decode_base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")  # type: ignore

#==================== Stream Video ======================
@app.get("/video-feed/face")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    _, jpeg = cv2.imencode(".jpg", latest_frame)
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # ~30 FPS

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# ====================== API ======================
ipcam_clients: List[WebSocket] = []

# === WebSocket ===
@app.websocket("/ws/face-detection")
async def websocket_face_detection(websocket: WebSocket):
    await websocket.accept()
    ipcam_clients.append(websocket)
    global latest_frame
    print(f"[INFO] Face API connected: {websocket.client}")

    try:
        while True:

            # API đợi dữ liệu
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

            # Nhận diện khuôn mặt
            faces = perform_recognize_face(frame)

            for face in faces:
                x1, y1, x2, y2 = face['box']
                name = face.get('name', 'Unknown')
                prob = face.get('probability', 0)

                # Nếu độ chính xác < 95%, đổi name thành [FACE]_Unknown
                if prob < 0.95:
                    name = "[FACE]_Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{name} ({int(prob*100)}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Cập nhật name lại vào dict (để gửi về frontend đúng)
                face['name'] = name

            # Cập nhật frame mới nhất
            with frame_lock:
                global latest_frame
                latest_frame = frame.copy()
                print("Updated latest frame")
            
            # Encode frame để gửi lại
            encoded_frame = encode_image_to_base64(frame)

            await websocket.send_text(json.dumps({
                "name": faces,
                "image_base64": encoded_frame # type: ignore
            }))

    except Exception as e:
        print(f"[ERROR] IPCam WebSocket: {e}")
        ipcam_clients.remove(websocket)
