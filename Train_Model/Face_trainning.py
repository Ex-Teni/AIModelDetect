import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from joblib import dump

from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# Cấu hình transform cho ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load model FaceNet (pretrained)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load dữ liệu từ thư mục dataset
dataset_path = "dataset"
embeddings = []
labels = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = facenet(img_tensor)
        embeddings.append(embedding.squeeze().cpu().numpy())
        labels.append(person_name)

# Mã hoá nhãn
le = LabelEncoder()
y = le.fit_transform(labels)
X = np.array(embeddings)

# Train model phân loại (SVM)
model = SVC(kernel="rbf", C=10, gamma=0.01, probability=True)
model.fit(X, y)

# Lưu model & label encoder
dump(model, 'face_classifier.joblib')
dump(le, 'label_encoder.joblib')

print("Training completed.")
