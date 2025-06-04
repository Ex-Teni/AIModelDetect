import base64
import json
import asyncio
import websockets
import cv2

# ===== Config =====
IMAGE_PATH = "test_image.jpg"
WEBSOCKET_ENDPOINTS = [
    "ws://localhost:8001/ws/plate-detection",
    "ws://localhost:8002/ws/container-detection",
    "ws://localhost:8003/ws/face-detection",
]

# ===== Helper =====
def encode_image_to_base64(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8") # type: ignore

# ===== Gửi ảnh tới nhiều WebSocket =====
async def send_image_to_all(base64_image, endpoints):
    async def send_to_one(uri):
        try:
            async with websockets.connect(uri, max_size=None) as websocket:
                print(f"[INFO] Connected to {uri}")
                payload = json.dumps({"image": base64_image})
                await websocket.send(payload)
                print(f"[INFO] Sent image to {uri}")

                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"[RESPONSE from {uri}]:\n{response}\n")
                except asyncio.TimeoutError:
                    print(f"[WARN] No response from {uri} within timeout.")
        except Exception as e:
            print(f"[ERROR] Failed to connect/send to {uri}: {e}")

    tasks = [send_to_one(uri) for uri in endpoints]
    await asyncio.gather(*tasks)

# ===== Main Entrypoint =====
async def main():
    base64_image = encode_image_to_base64(IMAGE_PATH)
    await send_image_to_all(base64_image, WEBSOCKET_ENDPOINTS)

if __name__ == "__main__":
    asyncio.run(main())
