import time
import cv2
import threading
import queue
from typing import Callable, List, Optional
from ..results import DetectionResult

class VideoProcessor:
    def __init__(self, 
                 source: str, 
                 detection_func: Callable, 
                 display: bool = False, 
                 save_path: str = None,
                 callback: Optional[Callable[[any, List['DetectionResult']], None]] = None,
                 draw_mode: str = 'all',
                 fast_mode: bool = True,
                 resize_display: Optional[tuple] = None,
                 ):
        """
        source: đường dẫn RTSP hoặc video file hoặc camera ID
        detection_func: hàm detect(frame) -> List[DetectionResult]
        display: có hiển thị realtime không
        save_path: nếu muốn lưu video output
        frame_skip: bỏ qua N frame để tăng tốc
        use_threading: sử dụng đa luồng để tách capture và detection
        max_queue_size: kích thước queue buffer (nhỏ hơn = ít lag hơn)
        resize_display: resize frame trước khi display (VD: (640, 480))
        show_fps: hiển thị FPS trên màn hình
        """
        self.source = source
        self.detection_func = detection_func
        self.display = display
        self.save_path = save_path
        self.callback = callback
        self.draw_mode = draw_mode
        self.fast_mode = fast_mode
        self.resize_display = resize_display
        self.stop_flag = False

    def _capture_thread(self, cap):
        """Thread để capture frame liên tục"""
        frame_id = 0
        while not self.stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                self.stop_flag.set()
                break
            
            frame_id += 1
            if frame_id % self.frame_skip != 0:
                continue
    
    def _detection_thread(self):
        """Thread riêng để chạy detection"""
        while not self.stop_flag.is_set():
            try:
                frame_id, frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Chạy detection
            detections = self.detection_func(frame)

            #call back nếu có
            if self.callback:
                try:
                    self.callback(frame, detections)
                except Exception as e:
                    print(f"[WARN] Data output error: {e}")

            # Đưa kết quả vào queue
            try:
                self.result_queue.put((frame_id, frame, detections), block=False)
            except queue.Full:
                # Bỏ kết quả cũ nhất
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put((frame_id, frame, detections), block=False)
                except:
                    pass
    
    def _draw_detections(self, frame, detections):
        """Vẽ kết quả detection lên frame"""
        for det in detections:
            if not det.text:
                continue
            x1, y1, x2, y2 = det.bbox

            # Vẽ bounding box
            if self.draw_mode in ["bbox", "all"] and not self.fast_mode:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Hiển thị text OCR
            if self.draw_mode in ["text", "all"]:
                label = f"{det.text}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def run(self):
        """Chạy video processing"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.source}")
            return

        fps_target = 30 if self.fast_mode else cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = 1.0 / fps_target

        writer = None
        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.save_path, fourcc, fps_target, (w, h))

        print(f"[INFO] Realtime video started (fast_mode={self.fast_mode}, draw_mode='{self.draw_mode}')")
        print("[INFO] Press '*' or 'ESC' to stop")

        last_time = time.time()
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # Giới hạn tốc độ hiển thị khi fast_mode
            if self.fast_mode:
                now = time.time()
                if now - last_time < frame_interval:
                    continue
                last_time = now

            detections = self.detection_func(frame)

            # Gửi kết quả OCR ra callback
            if self.callback:
                try:
                    self.callback(frame, detections)
                except Exception as e:
                    print(f"[WARN] Callback error: {e}")

            # Vẽ hiển thị
            if self.display:
                display_frame = self._draw_detections(frame.copy(), detections)
                if self.resize_display:
                    display_frame = cv2.resize(display_frame, self.resize_display)
                cv2.imshow("Realtime Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('*'), 27]:
                    break

            # Ghi video nếu cần
            if writer:
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Realtime processing stopped")

    def _run_single_thread(self, cap, writer):
        """Chạy single-thread mode"""
        frame_id = 0
        
        print("[INFO] Single-thread mode")
        print("[INFO] Press '*' or 'q' to stop")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            if frame_id % self.frame_skip != 0:
                if self.display:
                    cv2.imshow("Detection...", frame)
                    if cv2.waitKey(1) & 0xff in [ord('*'), 27]:
                        break
                continue
            
            # Detection
            detections = self.detection_func(frame)
            
            # Callback
            if self.callback:
                try:
                    self.callback(frame, detections)
                except Exception as e:
                    print(f"[WARN] Callback error: {e}")
            
            # Vẽ kết quả
            display_frame = self._draw_detections(frame.copy(), detections)
            
            # Display
            if self.display:
                if self.resize_display:
                    display_frame = cv2.resize(display_frame, self.resize_display)
                
                cv2.imshow("Realtime Detection", display_frame)
                if cv2.waitKey(1) & 0xFF in [ord('*'), ord('q'), 27]:
                    break
            
            # Save
            if writer:
                writer.write(frame)