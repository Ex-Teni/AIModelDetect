import os
import cv2
import base64
import requests
import numpy as np

# === Cấu hình ===
API_URL = "http://localhost:8000/process-batch"
IMAGE_FOLDER = "images/"
ALLOWED_EXTS = [".jpg", ".jpeg", ".png"]
RETURN_IMAGES = True

# === Mã hóa ảnh sang base64 ===
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8') # type: ignore

# === Giải mã base64 về ảnh OpenCV ===
def decode_image(b64_string):
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# === Gửi nhiều ảnh trong batch lên REST API ===
def send_batch(images_b64):
    payload = {
        "images": images_b64,
        "return_images": RETURN_IMAGES
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[ERROR] Status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[EXCEPTION] {e}")
    return None

# === Đọc tất cả ảnh từ folder và gửi ===
def main():
    # Lấy danh sách file ảnh
    files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
    files.sort()
    if not files:
        print("[ERROR] No images found in folder")
        return


    images = []
    filenames = []
    for filename in files:
        img_path = os.path.join(IMAGE_FOLDER, filename)
        image = cv2.imread(img_path)
        if image is not None:
            images.append(encode_image(image))
            filenames.append(filename)
        else:
            print(f"[WARNING] Can't read imaged: {img_path}")

    # Gửi batch ảnh
    result_data = send_batch(images)
    if not result_data or not result_data.get("results"):
        print("[ERROR] No responsive from server")
        return

    results = result_data["results"]

    # Hiển thị kết quả từng ảnh
    for i, result in enumerate(results):
        print(f"\n[IMAGE] {filenames[i]}")
        if result.get("success"):
            metadata = result.get("metadata", {})
            detections = result.get("detections", [])
            print(f"  - Metadata: {metadata}")
            for det in detections:
                print(f"    + {det['type'].upper()}: {det['text']} (conf={det['confidence']:.2f})")

            # Hiển thị ảnh annotate nếu có
            annotated_b64 = result.get("image")
            if annotated_b64:
                annotated_img = decode_image(annotated_b64)
                cv2.imshow(f"Result - {filenames[i]}", annotated_img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                cv2.destroyWindow(f"Result - {filenames[i]}")
        else:
            print(f"  - [ERROR] {result.get('error', 'Unknown error')}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
