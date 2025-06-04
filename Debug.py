# client_camera.py - FIXED VERSION
import asyncio
import json
import websockets
import base64
import cv2
import numpy as np

async def send_frames(uri):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    # Cấu hình camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        async with websockets.connect(uri) as websocket:
            print(f"[CONNECTED] Sending frames to {uri}")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Cannot read frame from webcam")
                    await asyncio.sleep(0.1)
                    continue

                # Resize nếu muốn (ví dụ resize về 640x480 để giảm băng thông)
                frame = cv2.resize(frame, (640, 480))
                
                # Encode frame → JPEG → base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8') # type: ignore
                
                # Gửi JSON chứa trường "image" - ĐÚNG FORMAT
                payload = json.dumps({"image": frame_b64})
                await websocket.send(payload)
                
                # Nhận phản hồi JSON từ server
                response = await websocket.recv()
                
                try:
                    data = json.loads(response)
                except json.JSONDecodeError as e:
                    print(f"[JSON ERROR] Cannot parse server response: {e}")
                    await asyncio.sleep(0.01)
                    continue
                
                # Kiểm tra server trả error
                if not data.get("success", False):
                    error_msg = data.get("error", "Unknown error")
                    print(f"[SERVER ERROR] {error_msg}")
                    await asyncio.sleep(0.01)
                    continue
                
                # Lấy ảnh đã annotate (base64) từ key "image"
                img_b64 = data.get("image")
                if not img_b64:
                    print("[WARNING] No annotated image received from server")
                    await asyncio.sleep(0.01)
                    continue
                
                try:
                    # Decode và hiển thị ảnh đã annotate
                    img_bytes = base64.b64decode(img_b64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    annotated = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if annotated is not None:
                        # Hiển thị trong cửa sổ OpenCV (để dev debug)
                        cv2.imshow("Annotated (Server)", annotated)
                        
                        # In detections ra console (mỗi 30 frame để tránh spam)
                        frame_count += 1
                        if frame_count % 30 == 0:
                            detections = data.get("detections", [])
                            metadata = data.get("metadata", {})
                            
                            print(f"\n[FRAME {frame_count}] Detections:")
                            for det in detections:
                                det_type = det.get('type', 'unknown')
                                text = det.get('text', 'None')
                                conf = det.get('confidence', 0.0)
                                print(f"  - {det_type.upper()}: {text} (conf={conf:.2f})")
                            
                            print(f"[METADATA] Plate: {metadata.get('plate', 'None')}, "
                                  f"Container: {metadata.get('container', 'None')}, "
                                  f"Face: {metadata.get('face', 'None')}")
                    
                except Exception as e:
                    print(f"[ERROR] Cannot decode annotated image: {e}")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quit requested by user")
                    break
                
                # Điều chỉnh FPS (30fps ~ 0.033s, 15fps ~ 0.067s)
                await asyncio.sleep(0.033)  # ~30 FPS
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"[ERROR] WebSocket connection closed: {e}")
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed")

if __name__ == "__main__":
    # WebSocket endpoint cho client camera
    WS_URI = "ws://localhost:8000/ws/combined-detection"
    
    print(f"[INFO] Starting camera client...")
    print(f"[INFO] Connecting to: {WS_URI}")
    print(f"[INFO] Press 'q' in the OpenCV window to quit")
    
    try:
        asyncio.run(send_frames(WS_URI))
    except KeyboardInterrupt:
        print("\n[INFO] Application terminated by user")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")