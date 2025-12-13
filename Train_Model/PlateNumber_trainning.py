from ultralytics import YOLO
import os

# Cấu hình dataset bằng file YAML
YAML_PATH = "dataset/data.yaml"
if not os.path.exists(YAML_PATH):
    raise FileNotFoundError(f"Không tìm thấy file: {YAML_PATH}")

# Chọn model YOLOv8 để huấn luyện
model_version = "yolov8s.pt"

# Số epoch huấn luyện + kích thước ảnh
epochs = 50
img_size = 640

# Load model Yolo
model = YOLO(model_version)

# Huấn luyện
model.train(
    data = YAML_PATH,
    epochs = epochs,
    imgsz = img_size,
    batch = 8,
    project = "results",
    name = "AI_MODEL",
    exist_ok = True,

    patience = 10,   # Early stop
    save = True,     # Lưu kết quả học đc
    lr0 = 0.001,     # Learning rate
    val = True,      # Đánh giá tập validation
    augment = True,  # Tự động thay đổi ảnh để học
    verbose = True,  # Hiển thị log chi tiết
)

print("Trainning complete!!!")