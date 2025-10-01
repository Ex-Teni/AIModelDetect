import cv2
from typing import Callable

class VideoProcessor:
    def __init__(self, source: str, detection_func: Callable, display: bool = True, save_path: str = None):
        """
        source: đường dẫn RTSP hoặc video file
        detection_func: hàm detect(frame) -> List[DetectionResult]
        display: có hiển thị realtime không
        save_path: nếu muốn lưu video output
        """
        self.source = source
        self.detection_func = detection_func
        self.display = display
        self.save_path = save_path

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.source}")
            return

        writer = None
        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.save_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detection_func(frame)

            # Vẽ kết quả
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{det.detection_type}: {det.text or ''} ({det.confidence:.2f})"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if self.display:
                cv2.imshow("Realtime Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if writer:
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
