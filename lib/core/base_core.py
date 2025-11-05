import numpy as np
import threading
from typing import Callable, List, Optional, Dict, Any, Union

from ..detectors import PlateDetector, ContainerDetector, FaceDetector
from ..results import DetectionResult
from ..utils import BatchProcessor, ImageProcessor, VideoProcessor

class MultiDetect_Core:
    """
    Lớp lõi cho hệ thống phát hiện đa đối tượng (biển số, container, khuôn mặt).
    Hỗ trợ:
      - Phát hiện + OCR theo từng loại
      - Xử lý ảnh theo batch
      - Xử lý video hoặc camera realtime (GPU / CPU)
      - Xuất kết quả ra nhiều định dạng (JSON, CSV, TXT)

    ----------------------------------------------------------
    CÁC HÀM PUBLIC ĐƯỢC SỬ DỤNG BÊN NGOÀI THƯ VIỆN (API CHÍNH)
    ----------------------------------------------------------

    1. Nhận diện đối tượng (Single & Combined)
        - detect_plates_only(image)
        - detect_containers_only(image)
        - detect_faces_only(image)
        - detect_all(image)

    2. Xử lý hàng loạt (Batch)
        - process_image_folder(folder_path, output_format="json")

    3. Xử lý video hoặc camera realtime
        - process_realtime_camera(
              source=0,
              mode="all",
              save_path=None,
              display=True,
              draw_mode="all",
              fast_mode=False,
              callback=None,
              resize_display=None
          )

    4. Xuất kết quả
        - export_results(detections, format_type="json")
    """

    def __init__(self, device: str = "auto"):
        self.device = device

        # Lazy loading
        self._plate_detector = None
        self._container_detector = None
        self._face_detector = None

        self.image_processor = ImageProcessor()
        self.batch_processor = BatchProcessor()
        
        self._lock = threading.Lock()
    
    @property
    def plate_detector(self) -> PlateDetector:
        """Lazy loading plate detector"""
        if self._plate_detector is None:
            with self._lock:
                if self._plate_detector is None:
                    self._plate_detector = PlateDetector(self.device)
        return self._plate_detector
    
    @property
    def container_detector(self) -> ContainerDetector:
        """Lazy loading container detector"""
        if self._container_detector is None:
            with self._lock:
                if self._container_detector is None:
                    self._container_detector = ContainerDetector(self.device)
        return self._container_detector
    
    @property  
    def face_detector(self) -> FaceDetector:
        """Lazy loading face detector"""
        if self._face_detector is None:
            with self._lock:
                if self._face_detector is None:
                    self._face_detector = FaceDetector(self.device)
        return self._face_detector
    

# ===== SINGLE DETECTION METHODS =====
    def detect_plates_only(self, image: np.ndarray) -> List[DetectionResult]:
        """
            Phát hiện & OCR biển số xe.

            VD: \n
                from lib import MultiDetect_Core
                core = MultiDetect_Core()
                image = cv2.imread("img001.jpg")
                results = core.detect_plates_only(image)

            Output (list DetectionResult): \n
                [
                  {
                    "type": "plate",
                    "text": "51F12345",
                    "bbox": [50, 60, 200, 120],
                    "confidence": 0.95,
                    "detection_confidence": 0.93,
                    "ocr_confidence": 0.91,
                    "failed_reason": "OK"
                  }
                ]
        """
        return self.plate_detector.detect(image)
    
    def detect_containers_only(self, image: np.ndarray) -> List[DetectionResult]:
        """
            Phát hiện & OCR container.

            VD:\n
                from lib import MultiDetect_Core
                core = MultiDetect_Core()
                image = cv2.imread("img001.jpg")
                detection = core.detect_containers_only(image)
            
            Output:\n
                [{
                  "type": "container",
                  "text": "CMAU1234567",
                  "bounding_box": [100, 200, 400, 250],
                  "confidence": 0.92,
                  "detection_confidence": 0.91,
                  "ocr_confidence": 0.89,
                  "failed_reason": "OK"
                }]
        """
        return self.container_detector.detect(image)
    
    def detect_faces_only(self, image: np.ndarray) -> List[DetectionResult]:
        """
            Phát hiện & nhận diện khuôn mặt

            VD:\n
                from lib import MultiDetect_Core
                core = MultiDetect_Core()
                image = cv2.imread("img001.jpg")
                detection = core.detect_faces_only(image)
            
            Output:\n
                [{
                  "type": "face",
                  "text": "Person_A",
                  "bounding_box": [120, 90, 200, 220],
                  "confidence": 0.98,
                  "detection_confidence": 0.96,
                  "recognition_confidence": 0.94,
                  "person_name": "Alice"
                }]
        """
        return self.face_detector.detect(image)
    
    
    # ===== COMBINED DETECTION METHODS =====
    def detect_all(self, image: np.ndarray) -> List[DetectionResult]:
        """
            Phát hiện & nhận diện cả 3 model cùng lúc

            VD:\n
                from lib import MultiDetect_Core
                core = MultiDetect_Core()
                image = cv2.imread("img001.jpg")
                detection = core.detect_all(image)
            
            Output: \n
                [
                  {
                    "type": "plate",
                    "text": "51F12345",
                    "bounding_box": [50, 60, 200, 120],
                    "confidence": 0.95,
                    "detection_confidence": 0.93,
                    "ocr_confidence": 0.91,
                    "is_multiline": false,
                    "failed_reason": "OK"
                  },
                  {
                    "type": "container",
                    "text": "CMAU1234567",
                    "bounding_box": [300, 150, 700, 250],
                    "confidence": 0.92,
                    "detection_confidence": 0.91,
                    "ocr_confidence": 0.89,
                    "failed_reason": "OK"
                  },
                  {
                    "type": "face",
                    "text": "Person_A",
                    "bounding_box": [800, 200, 950, 420],
                    "confidence": 0.97,
                    "detection_confidence": 0.96,
                    "recognition_confidence": 0.94,
                    "person_name": "Alice"
                  }
                ]
        """
        image_ref = image.copy()

        # Chạy song song để tăng tốc
        results: List[DetectionResult] = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs = {
                'plate': ex.submit(self.detect_plates_only, image_ref),
                'container': ex.submit(self.detect_containers_only, image_ref),
                'face': ex.submit(self.detect_faces_only, image_ref),
            }
            for name, f in futs.items():
                try:
                    res = f.result(timeout=5)
                    results.extend(res)
                except Exception as e:
                    print(f"[ERROR] {name} detect failed: {e}")
        return results
    
    
    # ===== BATCH PROCESSING METHODS =====
    def process_image_folder(self, folder_path: str, 
                           output_format: str = "json") -> Dict[str, Any]:
        """
        Xử lý toàn bộ ảnh trong 1 folder.
        
        VD:\n
            from lib import MultiDetect_Core
            results = core.process_image_folder("images/container", "detect_all",output_format="json")

        Output: \n
            {
              "total_images": 10,
              "successful_detections": 8,
              "results": {
                "img001.jpg": [{"type": "container", "text": "CMAU1234567", ...}],
                "img002.jpg": null
              }
            }
        """
        return self.batch_processor.process_folder(
            folder_path, self.detect_all, output_format
        )
    
    
    # ===== VIDEO PROCESSING METHODS =====
    def process_realtime_camera(self, 
                            source: Union[int, str] = 0,
                            mode: str = "all",
                            save_path: Optional[str] = None,
                            display: bool = True,
                            draw_mode: str = "all",      # "bbox", "text", "all"
                            fast_mode: bool = False,
                            callback: Optional[Callable] = None,
                            resize_display: Optional[tuple] = None): 
        
        """
        Xử lý video hoặc camera realtime (RTSP, file, webcam).

        Thông tin:
            core.process_realtime_camera(
                source="rtsp://192.168.1.10:554/live", -nguồn video, có thể là webcam (0), file video, hoặc RTSP \n
                mode="all",                            -loại đối tượng nhận diện ("plate", "container", "face", hoặc "all") \n
                save_path="output.mp4",                -nơi lưu video kết quả, để None nếu không cần lưu \n
                display=True,                          -hiển thị kết quả video cùng thời điểm code chạy \n
                draw_mode="all",                       -kiểu hiển thị ("bbox" chỉ khung, "text" chỉ OCR, "all" cả hai) \n
                fast_mode=True,                        -tăng tốc bằng cách bỏ frame trễ khi luồng video quá nhanh \n
                callback=on_detect,                    -trả kết quả 
                resize_display=(1280, 720)             - giảm độ phân giải khi hiển thị để tăng FPS \n
            )

        VD1 – chạy camera:
            core.process_realtime_camera(
                source=0,
                mode="plate",
                display=True,
                draw_mode="text",
                fast_mode=True
            )

        VD2 – chạy video file, lưu output:
            core.process_realtime_camera(
                source="input.mp4",
                mode="all",
                save_path="output.mp4",
                fast_mode=False,
                draw_mode="all"
            )

        VD3 – đọc luồng RTSP với callback:
            def on_detect(frame, detections):
                for d in detections:
                    print(d.text)

            core.process_realtime_camera(
                source="rtsp://192.168.1.10:554/stream",
                mode="container",
                callback=on_detect,
                draw_mode="text",
                display=True
            )
        """

        # Chọn hàm phát hiện tương ứng
        if mode == "plate":
            detect_func = self.detect_plates_only
        elif mode == "container":
            detect_func = self.detect_containers_only
        elif mode == "face":
            detect_func = self.detect_faces_only
        else:
            detect_func = self.detect_all

        processor = VideoProcessor(
            source=source,
            detection_func=detect_func,
            display=display,
            save_path=save_path,
            callback=callback,
            draw_mode=draw_mode,
            fast_mode=fast_mode,
            resize_display=resize_display
        )
        processor.run()
    
    # ===== UTILITY METHODS =====
    def export_results(self, detections: List[DetectionResult], 
                      format_type: str = "json") -> str:
        """
        Xuất kết quả nhận dạng ra file text, csv hoặc json.

        Ví dụ:
            detections = core.detect_all(image)
            json_text = core.export_results(detections, "json")
            csv_text  = core.export_results(detections, "csv")
            txt_text  = core.export_results(detections, "txt")

        Output (txt):
            PLATE: 51F12345 | Box: (50,60,200,120) | Confidence: 0.950
            CONTAINER: CMAU1234567 | Box: (100,200,400,250) | Confidence: 0.920
        """

        if format_type == "json":
            import json
            return json.dumps([d.to_dict() for d in detections], indent=2)
        
        elif format_type == "csv":
            import csv, io
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(['Type', 'Text', 'Bounding_Box', 'Confidence'])
            for d in detections:
                x1,y1,x2,y2 = d.bbox
                w.writerow([d.detection_type, d.text or 'None', f"({x1},{y1},{x2},{y2})", f"{d.confidence:.3f}"])
            return buf.getvalue()
        
        elif format_type == "txt":
            lines = []
            for d in detections:
                x1,y1,x2,y2 = d.bbox
                lines.append(f"{d.detection_type.upper()}: {d.text or 'None'} | Box: ({x1},{y1},{x2},{y2}) | Confidence: {d.confidence:.3f}")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
