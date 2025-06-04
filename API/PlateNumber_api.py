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
import easyocr

app = FastAPI()

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
plate_conf_threshold = 0.8  # Ngưỡng confidence cho biển số xe
ocr_conf_threshold = 0.2    # Ngưỡng confidence cho EasyOCR

# === Model AI ====
plate_model = YOLO("modelAI/detect_PlateNumber.pt") # model biển số xe
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())  # EasyOCR reader

# === Helper ===
def get_ocr_text(image, confidence_threshold=0.2):
    """
    Sử dụng EasyOCR để đọc text từ ảnh biển số xe
    Returns: (text, accuracy)
    """
    try:
        # Chuyển sang grayscale để OCR hoạt động tốt hơn
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Sử dụng EasyOCR để đọc text
        results = reader.readtext(gray)
        
        if not results:
            return "", 0.0
            
        # Nếu chỉ có 1 kết quả
        if len(results) == 1:
            bbox, text, confidence = results[0]
            if confidence >= confidence_threshold: # type: ignore
                return text.upper().replace(" ", ""), confidence
            else:
                return "", confidence
                
        # Nếu có nhiều kết quả, chọn kết quả tốt nhất
        best_result = max(results, key=lambda x: x[2])  # type: ignore # Chọn theo confidence cao nhất
        bbox, text, confidence = best_result
        
        if confidence >= confidence_threshold and len(text) >= 6:  # type: ignore # Biển số xe thường có ít nhất 6 ký tự
            return text.upper().replace(" ", ""), confidence
        else:
            # Thử kết hợp tất cả text nếu không có kết quả đủ tốt
            combined_text = "".join([result[1] for result in results if result[2] >= confidence_threshold]) # type: ignore
            if combined_text:
                avg_confidence = sum([result[2] for result in results if result[2] >= confidence_threshold]) / len([r for r in results if r[2] >= confidence_threshold]) # type: ignore
                return combined_text.upper().replace(" ", ""), avg_confidence
            
        return "", 0.0
        
    except Exception as e:
        print(f"[ERROR] OCR processing failed: {e}")
        return "", 0.0

def decode_base64_to_image(base64_str): # Nhận video stream dưới dạng base64 đưa lại thành ảnh
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image_to_base64(image): # Gửi video strem dưới dạng ảnh đưa lại thành base64
    _, buffer = cv2.imencode(".jpg" ,image)
    return base64.b64encode(buffer).decode("utf-8") # type: ignore

def preprocess_plate_image(plate_image):
    """
    Tiền xử lý ảnh biển số để OCR hoạt động tốt hơn
    """
    # Resize để có kích thước phù hợp
    height, width = plate_image.shape[:2]
    if height < 50:
        scale = 50 / height
        new_width = int(width * scale)
        plate_image = cv2.resize(plate_image, (new_width, 50))
    
    # Áp dụng một số kỹ thuật xử lý ảnh để cải thiện OCR
    # Chuyển sang grayscale
    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image
    
    # Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur nhẹ để giảm noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred

#==================== Stream Video ======================
latest_frame = None
frame_lock = threading.Lock()

@app.get("/video-feed/plate-detection")
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

@app.get("/start-stream")
def start_stream():
    subprocess.Popen(["python", "client_camera.py"])
    return {"status": "streaming started"}

# ======================== API ==========================

# === WebSocket ===
@app.websocket("/ws/plate-detection")
async def websocket_plate_detection(websocket: WebSocket):
    await websocket.accept()
    global latest_frame
    print(f"[INFO] PlateNumber API connected: {websocket.client}")

    while True:
        try:
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
            plate_result = plate_model(frame)[0]

            detection_result = []

            # Vẽ bouding box 
            for plate_box in plate_result.boxes:
                plate_conf = float(plate_box.conf[0])
                x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
                
                # Crop biển số từ frame gốc
                cropped_plate = frame[y1:y2, x1:x2]

                if plate_conf < plate_conf_threshold: # Biển số xe <80% thì gắn nhãn [PLATE]_Unknown
                    label_text = "[PLATE]_Unknown"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label_text, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    detection_result.append({
                        "box": [x1, y1, x2, y2],
                        "plate": "[PLATE]_Unknown"
                    })
                    continue

                # Nếu plate_conf >= 0.8 thì sử dụng EasyOCR để đọc text
                # Tiền xử lý ảnh biển số
                processed_plate = preprocess_plate_image(cropped_plate)
                
                # Sử dụng EasyOCR để đọc text
                recognized_plate_text, ocr_confidence = get_ocr_text(processed_plate, ocr_conf_threshold)
                
                # Tính accuracy dựa trên confidence của OCR
                accuracy = ocr_confidence

                # Vẽ bounding box và plate text lên frame
                if accuracy >= 0.5:  # type: ignore # Ngưỡng để hiển thị kết quả
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{recognized_plate_text} ({accuracy*100:.1f}%)"
                    cv2.putText(frame, label_text, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    detection_result.append({
                        "box": [x1, y1, x2, y2],
                        "plate": recognized_plate_text
                    })
                else:
                    # Nếu OCR không đủ tin cậy
                    label_text = "[PLATE]_Low_Confidence"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label_text, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    detection_result.append({
                        "box": [x1, y1, x2, y2],
                        "plate": "[PLATE]_Low_Confidence"
                    })
            
            # Cập nhật frame mới nhất để stream qua /video-feed
            with frame_lock:
                latest_frame = frame.copy()

            # Encode frame để gửi lại
            encoded_frame = encode_image_to_base64(frame)

            await websocket.send_text(json.dumps({
                "plates": detection_result,
                "image_base64": encoded_frame
            }, ensure_ascii=False))

        except Exception as e:
            print("Error:", e)
            break