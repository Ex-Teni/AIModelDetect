import numpy as np
import threading
from typing import Callable, List, Optional, Dict, Any, Union

from ..detectors import PlateDetector, ContainerDetector, FaceDetector
from ..results import DetectionResult
from ..utils import BatchProcessor, ImageProcessor, VideoProcessor

class MultiDetect_Core:
    """
    Thư viện phát hiện đa đối tượng chính
    Hỗ trợ phát hiện biển số, container code và khuôn mặt
    """
    def __init__(self, device: str = "auto"):
        """
        detector_plate: mô hình YOLO detect biển số
        detector_container: mô hình YOLO detect container
        ocr_plate: module OCR biển số (PlateOCR)
        ocr_container: module OCR container (ContainerOCR)
        """
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

            VD:\n
            from lib import MultiDetect_Core\n
            core = MultiDetect_Core()\n
            image = cv2.imread("img001.jpg")\n
            detection = core.plates_only(image)\n
            
            Output:\n
            [{
              "type": "plate",
              "text": "51F12345",
              "bounding_box": [50, 60, 200, 120],
              "confidence": 0.95,
              "detection_confidence": 0.93,
              "ocr_confidence": 0.91,
              "is_multiline": false,
              "failed_reason": "OK"
            }]
        """
        return self.plate_detector.detect(image)
    
    def detect_containers_only(self, image: np.ndarray) -> List[DetectionResult]:
        """
            Phát hiện & OCR container.

            VD:\n
            from lib import MultiDetect_Core\n
            core = MultiDetect_Core()\n
            image = cv2.imread("img001.jpg")\n
            detection = core.detect_containers_only(image)\n
            
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
            from lib import MultiDetect_Core\n
            core = MultiDetect_Core()\n
            image = cv2.imread("img001.jpg")\n
            detection = core.detect_faces_only(image)\n
            
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
            Phát hiện & nhận diện khuôn mặt

            VD:\n
            from lib import MultiDetect_Core\n
            core = MultiDetect_Core()\n
            image = cv2.imread("img001.jpg")\n
            detection = core.detect_all(image)\n
            
            Output:
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
        from lib import MultiDetect_Core\n
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
        Xử lý camera/video realtime với tối ưu hiệu năng

        VD:
            # Camera realtime với tối ưu cao
            core.process_realtime_camera(
                source=0,
                mode="plate",
                frame_skip=2,           # Xử lý mỗi 2 frame
                use_threading=True,      # Bật multi-threading
                max_queue_size=2,        # Buffer nhỏ để giảm lag
                resize_display=(960, 540),  # Resize display
                show_fps=True
            )

            # Video file với chất lượng tốt nhất
            core.process_realtime_camera(
                source="video.mp4",
                mode="all",
                save_path="output.mp4",
                frame_skip=1,
                use_threading=True
            )

            # RTSP stream với callback
            def my_callback(frame, detections):
                print(f"Detected {len(detections)} objects")

            core.process_realtime_camera(
                source="rtsp://camera_ip:554/stream",
                mode="container",
                callback=my_callback,
                frame_skip=3,
                use_threading=True,
                max_queue_size=2
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
            Export kết quả ra nhiều format
        
            VD:\n
            from lib import MultiDetect_Core\n
            detections = core.detect_all(image)\n
            json_f = core.export_results(detections, format_type="json"))\n
            csv_f = core.export_results(detections, format_type="csv"))\n
            txt_f = core.export_results(detections, format_type="txt"))\n
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
        
