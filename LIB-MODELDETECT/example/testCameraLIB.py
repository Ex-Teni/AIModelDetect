import csv
from datetime import datetime
from lib import MultiDetect_Core

CSV_PATH = "realtime_result.csv"
frame_counter = 0


def write_csv_header(csv_path: str):
    """Tạo file CSV mới và ghi header."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Timestamp", "License Plate", "Container Code"])


def append_detection_to_csv(csv_path: str, plates, containers):
    """Ghi 1 dòng kết quả vào CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            ", ".join(plates) if plates else "",
            ", ".join(containers) if containers else ""
        ])


def detection_callback(frame, detections):
    """
    Callback được gọi sau mỗi frame detect.
    - detections: danh sách DetectionResult
    - frame: ảnh gốc (đã được xử lý nếu cần)
    """
    global frame_counter
    frame_counter += 1

    plates, containers = [], []
    for det in detections:
        if det.detection_type == "plate" and det.text:
            plates.append(det.text)
        elif det.detection_type == "container" and det.text:
            containers.append(det.text)

    if plates or containers:
        append_detection_to_csv(CSV_PATH, plates, containers)


def test_realtime_video(source: str,
                        mode="all",
                        device="cuda",
                        save_path=None,
                        display=True,
                        csv_path: str = CSV_PATH,
                        draw_mode="all",
                        ):
    """
    Chạy realtime detect từ video .mp4 hoặc camera bằng MultiDetect_Core.
    Ghi kết quả biển số và container vào CSV.
    """
    core = MultiDetect_Core(device=device)
    write_csv_header(csv_path)

    print(f"[*] Starting realtime detection from: {source}")
    print(f"[*] Output CSV: {csv_path}")
    print("Press 'ESC' to stop.\n")

    # Kích hoạt chế độ fast_mode (chỉ hiển thị OCR, không vẽ bbox)
    core.process_realtime_camera(
        source=source,
        mode=mode,
        save_path=save_path,
        display=display,
        draw_mode=draw_mode,     # Thêm chế độ hiển thị text-only
        callback=detection_callback,
    )

    print(f"\n[INFO] Detection finished. Results saved to: {csv_path}")


if __name__ == "__main__":
    video_path = "example/images/camera/video.mp4"
    test_realtime_video(
        source=video_path,
        mode="all",
        device="cuda",
        display=True,
        draw_mode="all",  # Chỉ hiển thị OCR text, không vẽ bbox
        csv_path="example/output/camera/realtime_result.csv"
    )
