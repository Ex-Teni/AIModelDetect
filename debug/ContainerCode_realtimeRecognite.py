import os
import cv2
import torch
from fastapi import FastAPI, Query
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from typing import List
import shutil

# === Config ===
alphabet = sorted("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
conf_thresh = 0.4  # Ngưỡng confidence

# === Load models ===
ctncode_model = YOLO("modelAI/detect_ContainerCode.pt")
char_model = YOLO("modelAI/detect_Character.pt")

# === Mapping label ID sang ký tự ===
char2idx = {c: i + 1 for i, c in enumerate(alphabet)}
idx2char = {i: c for c, i in char2idx.items()}
idx2char[0] = ""  # blank

# === Khởi tạo app ===
app = FastAPI(title="Container Code Detect API")

# === Hàm nhóm bounding box thành dòng và sắp xếp trái qua phải ===
def group_boxes(boxes, y_thresh=20):
    rows = []
    boxes = sorted(boxes, key=lambda b: b[1])  # sort by y1
    for box in boxes:
        placed = False
        for row in rows:
            if abs(box[1] - row[0][1]) < y_thresh:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    for row in rows:
        row.sort(key=lambda b: b[0])  # sort by x1
    rows.sort(key=lambda r: r[0][1])  # sort rows by y
    return rows


cap = cv2.VideoCapture(0)  # Link IP camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    plate_results = ctncode_model(frame)[0]

    for plate_box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
        plate_crop = cv2.resize(frame[y1:y2, x1:x2], (320, 80))
        char_results = char_model(plate_crop)[0]

        chars = []
        for cbox in char_results.boxes:
            if cbox.conf[0] < conf_thresh:
                continue
            cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0])
            cls_id = int(cbox.cls[0])
            char = idx2char.get(cls_id, '?')
            if char not in alphabet:
                continue
            chars.append([cx1, cy1, cx2, cy2, char])

        lines = group_boxes(chars)
        full_plate = ''.join(c[4] for line in lines for c in line)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, full_plate, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.imshow("Realtime Container Code Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()