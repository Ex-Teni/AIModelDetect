import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from joblib import load
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
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

