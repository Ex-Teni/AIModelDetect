import asyncio
import json
import websockets
import base64
import cv2
import numpy as np
import os
import argparse

async def send_frames(uri, source_type, folder_path=None):
    frame_count = 0

    if source_type == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    elif source_type == "folder":
        if not folder_path or not os.path.exists(folder_path):
            print(f"[ERROR] Invalid folder path: {folder_path}")
            return
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not image_files:
            print(f"[ERROR] No images found in folder: {folder_path}")
            return
    else:
        print("[ERROR] Unknown source type. Use 'webcam' or 'folder'")
        return

    try:
        async with websockets.connect(uri) as websocket:
            print(f"[CONNECTED] Sending {'webcam frames' if source_type == 'webcam' else 'images'} to {uri}")
            
            while True:
                if source_type == "webcam":
                    ret, frame = cap.read()
                    if not ret:
                        print("[WARNING] Cannot read frame from webcam")
                        await asyncio.sleep(0.1)
                        continue
                else:  # folder
                    if frame_count >= len(image_files):
                        print("[INFO] All images sent.")
                        break
                    image_path = os.path.join(folder_path, image_files[frame_count]) # type: ignore
                    frame = cv2.imread(image_path)
                    if frame is None:
                        print(f"[WARNING] Cannot read image: {image_path}")
                        frame_count += 1
                        continue

                # Resize frame
                frame = cv2.resize(frame, (640, 480))

                # Encode to JPEG → base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8') # type: ignore
                payload = json.dumps({"image": frame_b64})
                await websocket.send(payload)

                # Receive and handle server response
                response = await websocket.recv()
                try:
                    data = json.loads(response)
                except json.JSONDecodeError as e:
                    print(f"[JSON ERROR] Cannot parse server response: {e}")
                    await asyncio.sleep(0.01)
                    continue

                if not data.get("success", False):
                    print(f"[SERVER ERROR] {data.get('error', 'Unknown error')}")
                    await asyncio.sleep(0.01)
                    continue

                img_b64 = data.get("image")
                if img_b64:
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        annotated = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if annotated is not None:
                            cv2.imshow("Annotated (Server)", annotated)

                            if frame_count % 10 == 0:
                                detections = data.get("detections", [])
                                metadata = data.get("metadata", {})
                                print(f"\n[FRAME {frame_count}] Detections:")
                                for det in detections:
                                    print(f"  - {det.get('type', 'unknown').upper()}: {det.get('text', 'None')} (conf={det.get('confidence', 0.0):.2f})")
                                print(f"[METADATA] Plate: {metadata.get('plate', 'None')}, "
                                      f"Container: {metadata.get('container', 'None')}, "
                                      f"Face: {metadata.get('face', 'None')}")

                    except Exception as e:
                        print(f"[ERROR] Cannot decode annotated image: {e}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quit requested by user")
                    break

                frame_count += 1
                await asyncio.sleep(0.1 if source_type == "folder" else 0.033)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"[ERROR] WebSocket connection closed: {e}")
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        if source_type == "webcam":
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client camera sender")
    parser.add_argument("--source", choices=["webcam", "folder"], required=True, help="Input source type")
    parser.add_argument("--folder", type=str, help="Folder path (required if --source folder)")
    parser.add_argument("--uri", type=str, default="ws://localhost:8000/ws/combined-detection", help="WebSocket URI")

    args = parser.parse_args()

    if args.source == "folder" and not args.folder:
        print("[ERROR] --folder path is required when using source=folder")
        exit(1)

    print(f"[INFO] Starting client with source: {args.source}")
    asyncio.run(send_frames(args.uri, args.source, args.folder))
