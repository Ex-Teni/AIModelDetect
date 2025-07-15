import os
import cv2
import json
import base64
import argparse
import requests
from time import time, sleep

def encode_image(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode() # type: ignore

def send_image_rest(api_url: str, img_b64: str) -> dict:
    payload = {"images": [img_b64]}
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        if response.ok:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def draw_detections(image, detections):
    colors = {
        "plate": (0, 255, 0),
        "container": (255, 0, 0),
        "face": (0, 0, 255),
        "seal": (255, 255, 0)
    }
    for det in detections:
        x1, y1, x2, y2 = det.get("box", [0, 0, 0, 0])
        det_type = det.get("type", "unknown")
        text = det.get("text", "None")
        color = colors.get(det_type, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{det_type}: {text}"
        cv2.putText(image, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main(folder_path: str, api_url: str, delay_between_images: float = 0.2):
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"[INFO] Found {len(image_files)} images. Sending one by one via REST...")

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        try:
            img_b64 = encode_image(image_path)
            print(f"\n[PROCESSING] {filename}")
            start = time()
            result = send_image_rest(api_url, img_b64)
            duration = round(time() - start, 2)

            if result.get("success"):
                detection = result["results"][0]
                metadata = detection.get("metadata", {})
                detections = detection.get("detections", [])
                print(f"[RESULT] Metadata: {metadata} | Time: {duration}s")

                # Hiển thị ảnh kèm bounding box
                original_img = cv2.imread(image_path)
                if original_img is not None:
                    annotated = draw_detections(original_img.copy(), detections)
                    cv2.imshow("Result", annotated)

                    # Tự động chuyển sau delay, hoặc nhấn 'q' để thoát
                    key = cv2.waitKey(int(delay_between_images * 1000)) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Quit requested.")
                        break

            else:
                print(f"[ERROR] Failed to process {filename}: {result.get('error')}")

        except Exception as e:
            print(f"[EXCEPTION] {filename}: {e}")

    cv2.destroyAllWindows()
    print("[INFO] All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto REST Client for Folder Images")
    parser.add_argument("--folder", type=str, required=True, help="Folder chứa ảnh cần gửi")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/process-batch", help="REST API URL")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay giữa các ảnh (giây)")

    args = parser.parse_args()
    main(args.folder, args.api_url, args.delay)
