import base64
import json
import threading
import time
import cv2
from fastapi.responses import StreamingResponse
import numpy as np
from fastapi import FastAPI, WebSocket
import torch
import subprocess
import easyocr

app = FastAPI()

# ========== Config ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
ocr_conf_threshold = 0.3        # Ngưỡng confidence cho EasyOCR
min_text_length = 2             # Độ dài tối thiểu của text để hiển thị

# ========== EasyOCR Reader ==========
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())  # EasyOCR reader

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


def get_all_text_from_image(image, confidence_threshold=0.3):
    """
    Sử dụng EasyOCR để đọc tất cả text từ ảnh
    Returns: list of (bbox, text, confidence)
    """
    try:
        # Chuyển sang grayscale để OCR hoạt động tốt hơn
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Sử dụng EasyOCR để đọc text
        results = reader.readtext(gray)
        
        # Lọc kết quả theo confidence và độ dài
        filtered_results = []
        for result in results:
            bbox, text, confidence = result
            if confidence >= confidence_threshold and len(text.strip()) >= min_text_length: # type: ignore
                # Chuyển bbox từ format của EasyOCR sang format [x1, y1, x2, y2]
                bbox_points = np.array(bbox)
                x1 = int(np.min(bbox_points[:, 0]))
                y1 = int(np.min(bbox_points[:, 1]))
                x2 = int(np.max(bbox_points[:, 0]))
                y2 = int(np.max(bbox_points[:, 1]))
                
                filtered_results.append({
                    "bbox": [x1, y1, x2, y2],
                    "text": text.strip(),
                    "confidence": confidence
                })
        
        return filtered_results
        
    except Exception as e:
        print(f"[ERROR] OCR processing failed: {e}")
        return []


def preprocess_image_for_ocr(image):
    # Tiền xử lý ảnh để OCR hoạt động tốt hơn
    # Chuyển sang grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur nhẹ để giảm noise
    blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
    
    return blurred


def draw_text_boxes(image, text_results):
    """
    Vẽ bounding box và text lên ảnh
    """
    for result in text_results:
        bbox = result["bbox"]
        text = result["text"]
        confidence = result["confidence"]
        
        x1, y1, x2, y2 = bbox
        
        # Chọn màu dựa trên confidence
        if confidence >= 0.8:
            color = (0, 255, 0)  # Xanh lá - confidence cao
        elif confidence >= 0.5:
            color = (0, 165, 255)  # Cam - confidence trung bình
        else:
            color = (0, 0, 255)  # Đỏ - confidence thấp
        
        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ text và confidence
        label_text = f"{text} ({confidence*100:.1f}%)"
        
        # Tính toán vị trí text để không bị che
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = max(y1 - 10, label_size[1] + 10)
        
        # Vẽ background cho text
        cv2.rectangle(image, (x1, label_y - label_size[1] - 10), 
                     (x1 + label_size[0], label_y + 5), color, -1)
        
        # Vẽ text
        cv2.putText(image, label_text, (x1, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


# ========== Streaming Endpoint ==========
@app.get("/video-feed/text-detection")
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
@app.websocket("/ws/text-detection")
async def websocket_text_detection(websocket: WebSocket):
    await websocket.accept()
    global latest_frame
    print(f"[INFO] Text OCR API connected: {websocket.client}")

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
            original_frame = frame.copy()
            
            # Tiền xử lý ảnh để OCR tốt hơn
            processed_frame = preprocess_image_for_ocr(frame)
            
            # Sử dụng EasyOCR để đọc tất cả text trong ảnh
            text_results = get_all_text_from_image(processed_frame, ocr_conf_threshold)
            
            # Vẽ bounding box và text lên frame gốc
            annotated_frame = draw_text_boxes(original_frame, text_results)
            
            # Tạo response data
            detection_results = []
            for result in text_results:
                detection_results.append({
                    "box": result["bbox"],
                    "text": result["text"],
                    "confidence": result["confidence"]
                })
            
            # Thêm thông tin tổng quan
            total_texts = len(detection_results)
            high_conf_texts = len([r for r in detection_results if r["confidence"] >= 0.8])
            
            # Vẽ thông tin tổng quan lên góc trái trên
            info_text = f"Total: {total_texts} | High Conf: {high_conf_texts}"
            cv2.rectangle(annotated_frame, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(annotated_frame, info_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Cập nhật frame
            with frame_lock:
                latest_frame = annotated_frame.copy()

            # Gửi kết quả qua WebSocket
            encoded_frame = encode_image_to_base64(annotated_frame)
            await websocket.send_text(json.dumps({
                "texts": detection_results,
                "total_count": total_texts,
                "high_confidence_count": high_conf_texts,
                "image_base64": encoded_frame
            }, ensure_ascii=False))

        except Exception as e:
            print("[ERROR] WebSocket Error:", e)
            break


# ========== Additional Endpoints ==========
@app.get("/")
def read_root():
    return {
        "message": "Text OCR Detection API",
        "endpoints": {
            "websocket": "/ws/text-detection",
            "video_stream": "/video-feed/text-detection",
            "start_stream": "/start-stream"
        },
        "config": {
            "ocr_confidence_threshold": ocr_conf_threshold,
            "min_text_length": min_text_length,
            "device": device
        }
    }


@app.get("/config")
def get_config():
    return {
        "ocr_conf_threshold": ocr_conf_threshold,
        "min_text_length": min_text_length,
        "device": device
    }


@app.post("/config")
async def update_config(config_data: dict):
    global ocr_conf_threshold, min_text_length
    
    if "ocr_conf_threshold" in config_data:
        ocr_conf_threshold = float(config_data["ocr_conf_threshold"])
    
    if "min_text_length" in config_data:
        min_text_length = int(config_data["min_text_length"])
    
    return {
        "message": "Config updated successfully",
        "new_config": {
            "ocr_conf_threshold": ocr_conf_threshold,
            "min_text_length": min_text_length
        }
    }