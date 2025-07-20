import os
import requests
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000/process-batch"
FOLDER_PATH = "images/"  

def send_image_to_api(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "images": [img_b64]
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("success") and result.get("results"):
                return result["results"][0]  # Chỉ gửi 1 ảnh, lấy kết quả đầu
        else:
            print(f"[ERROR] API returned status {response.status_code}")
    except Exception as e:
        print(f"[EXCEPTION] {e}")
    return None

def draw_results(image_path, results):
    image = cv2.imread(image_path)
    if image is None or not results or "detections" not in results:
        return image

    for det in results["detections"]:
        box = det["box"]
        text = det.get("text", "")
        conf = det.get("confidence", 0)

        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{det['type'].upper()}: {text} ({conf:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

def show_image(image, title="Annotated Result"):
    if image is None:
        print("[WARNING] Cannot display empty image")
        return
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.pause(2)  
    plt.close()

def main():
    image_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(FOLDER_PATH, filename)
        print(f"[{idx}/{len(image_files)}] Sending {filename} to API...")

        results = send_image_to_api(image_path)
        if results is None:
            print("[SKIP] Failed to get result from API.")
            continue

        annotated_img = draw_results(image_path, results)
        show_image(annotated_img, title=filename)

if __name__ == "__main__":
    main()
