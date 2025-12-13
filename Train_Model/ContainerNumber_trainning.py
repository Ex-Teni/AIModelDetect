from ultralytics import YOLO
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import time
from IPython.display import clear_output

# Cấu hình dataset bằng file YAML
YAML_PATH = "dataset/data.yaml"
if not os.path.exists(YAML_PATH):
    raise FileNotFoundError(f"Không tìm thấy file: {YAML_PATH}")

# Chọn model YOLOv8 để huấn luyện
model_version = "yolov8s.pt"

# Số epoch huấn luyện + kích thước ảnh
epochs = 80
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


log_path = 'results/AI_MODEL/results.csv'
last_epoch = -1

print("Bắt đầu theo dõi quá trình huấn luyện...")

while True:
    if not os.path.exists(log_path):
        print("Chưa thấy file log. Chờ 10 giây rồi thử lại...")
        time.sleep(10)
        continue

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print("Lỗi khi đọc log:", e)
        time.sleep(10)
        continue

    if len(df) == 0 or df.index[-1] == last_epoch:
        time.sleep(10)
        continue

    last_epoch = df.index[-1]

    # Tạo cột epoch từ index (nếu chưa có sẵn)
    df['epoch'] = df.index + 1

    # === Phân tích các chỉ số ===
    box_loss = df['train/box_loss']
    cls_loss = df['train/cls_loss']
    map50 = df['metrics/mAP50(B)']
    epochs = df['epoch']

    clear_output(wait=True)
    print(f"Epoch hiện tại: {last_epoch + 1} / {df['epoch'].max()}")
    print(f"Box Loss mới nhất: {box_loss.iloc[-1]:.4f}")
    print(f"mAP@50 mới nhất: {map50.iloc[-1]:.4f}")

    # === Kiểm tra overfitting hoặc giảm chất lượng ===
    if len(map50) > 3:
        if map50.iloc[-1] < map50.iloc[-2] < map50.iloc[-3]:
            print("[WARNING] Cảnh báo: mAP@50 đang giảm liên tiếp → mô hình có thể bị overfitting.")
        if box_loss.iloc[-1] > box_loss.min() * 1.5:
            print("[WARNING] Cảnh báo: Box loss tăng cao → kiểm tra learning rate hoặc augment.")

    # === Vẽ biểu đồ ===
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, box_loss, label='Box Loss')
    plt.plot(epochs, cls_loss, label='Cls Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, map50, label='mAP@50', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50")
    plt.title("Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    time.sleep(30)  # cập nhật mỗi 30 giây
