import asyncio
import json
import websockets
import base64
import cv2
import numpy as np
import threading

# cap = cv2.VideoCapture("rtsp://admin:12345@192.168.1.100:554/Streaming/Channels/101", cv2.CAP_FFMPEG) # Thay thế link rtsp
cap = cv2.VideoCapture(0)

lock = threading.Lock()
latest_frame = None  # Dùng để cập nhật frame hiển thị

# ================= GUI hiển thị frame ====================
def display_loop():
    global latest_frame
    while True:
        with lock:
            if latest_frame is not None:
                cv2.imshow("All Detections", latest_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ================= Gửi + nhận WebSocket ====================
async def send_and_receive(uri, label):
    global latest_frame
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print(f"[CONNECTED] {label} Connected to {uri}")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("[WARNING] Không đọc được frame từ camera")
                        continue

                    # Encode frame -> base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_encoded = base64.b64encode(buffer).decode('utf-8') # type: ignore

                    # Gửi ảnh
                    await websocket.send(json.dumps({"image": frame_encoded}))

                    # Nhận phản hồi
                    response = await websocket.recv()
                    data = json.loads(response)

                    # Decode ảnh kết quả (đã có bounding box)
                    frame_bytes = base64.b64decode(data['image_base64'])
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    with lock:
                        latest_frame = annotated_frame.copy()

                    # In ra kết quả nếu có
                    if label == "Plate" and data.get("plates"):
                        print(f"[{label}] Plate(s): {data['plates']}")

                    elif label == "Face" and data.get("name"):
                        print(f"[{label}] Face(s): {data['name']}")

                    elif label == "Container" and data.get("containers"):
                        print(f"[{label}] Container(s): {data['containers']}")

        except Exception as e:
            print(f"[ERROR] {label} Error: {e}, reconnecting in 5s ...")
            await asyncio.sleep(20)

# ================= MAIN ====================
async def main():
    tasks = [
        asyncio.create_task(send_and_receive('ws://localhost:8001/ws/plate-detection', 'Plate')),
        asyncio.create_task(send_and_receive('ws://localhost:8002/ws/container-detection', 'Container')),
        asyncio.create_task(send_and_receive('ws://localhost:8003/ws/face-detection', 'Face')),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Bắt đầu luồng hiển thị GUI
    display_thread = threading.Thread(target=display_loop, daemon=True)
    display_thread.start()

    # Chạy vòng lặp gửi ảnh và nhận kết quả
    asyncio.run(main())
