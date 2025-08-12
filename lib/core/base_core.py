import cv2
import numpy as np
import threading
from typing import List, Optional, Dict, Any, Callable

from ..detectors import PlateDetector, ContainerDetector, FaceDetector
from ..results import DetectionResult
from ..utils import BatchProcessor, ImageProcessor

class MultiDetect_Core:
    """
    Thư viện phát hiện đa đối tượng chính
    Hỗ trợ phát hiện biển số, container code và khuôn mặt
    """
    def __init__(self, device: str = "auto"):
        """
        Khởi tạo thư viện
        
        Args:
            model_path: Đường dẫn đến thư mục chứa models
            device: 'cpu', 'cuda', hoặc 'auto'
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
        """Chỉ phát hiện biển số xe"""
        return self.plate_detector.detect(image)
    
    def detect_containers_only(self, image: np.ndarray) -> List[DetectionResult]:
        """Chỉ phát hiện container code"""
        return self.container_detector.detect(image)
    
    def detect_faces_only(self, image: np.ndarray) -> List[DetectionResult]:
        """Chỉ phát hiện khuôn mặt"""
        return self.face_detector.detect(image)
    
    # ===== COMBINED DETECTION METHODS =====
    
    def detect_all(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Phát hiện tất cả đối tượng trong ảnh
        
        Args:
            image: Ảnh đầu vào (BGR format)
            
        Returns:
            List[DetectionResult]: Danh sách kết quả phát hiện
        """
        
        # Chạy song song để tăng tốc
        results: List[DetectionResult] = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs = [
                ex.submit(self.detect_plates_only, image),
                ex.submit(self.detect_containers_only, image),
                ex.submit(self.detect_faces_only, image),
            ]
            for f in futs:
                try:
                    results.extend(f.result())
                except Exception as e:
                    print(f"[ERROR] task failed: {e}")
        return results
    
    # ===== BATCH PROCESSING METHODS =====
    
    def process_image_folder(self, folder_path: str, 
                           output_format: str = "json") -> Dict[str, Any]:
        """
        Xử lý toàn bộ thư mục ảnh
        
        Args:
            folder_path: Đường dẫn thư mục ảnh
            output_format: "json", "text", "csv"
        """
        return self.batch_processor.process_folder(
            folder_path, self.detect_all, output_format
        )
    
    # ===== VIDEO PROCESSING METHODS =====
    
    # def process_realtime_camera(self, camera_id: int = 0, 
    #                            callback_func: Optional[Callable[[np.ndarray, List[DetectionResult]], None]] = None): 
    #     """
    #     Xử lý camera realtime
        
    #     Args:
    #         camera_id: ID camera
    #         callback_func: Hàm callback nhận kết quả mỗi frame
    #     """
    #     cap = cv2.VideoCapture(camera_id)
        
    #     try:
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
                
    #             # Phát hiện đối tượng
    #             detections = self.detect_all(frame)
                
    #             # Vẽ annotation
    #             annotated_frame = self.image_processor.draw_detections(frame, detections)
                
    #             # Gọi callback nếu có
    #             if callback_func:
    #                 callback_func(annotated_frame, detections)
                
    #             # Hiển thị
    #             cv2.imshow('MultiDetect Live', annotated_frame)
                
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break
    #     finally:
    #         cap.release()
    #         cv2.destroyAllWindows()
    
    # ===== UTILITY METHODS =====
    
    def get_text_summary(self, detections: List[DetectionResult]) -> Dict[str, List[str]]:
        """
        Trả về tóm tắt dạng text của tất cả detections
        
        Returns:
            Dict với key là loại object, value là list text detected
        """
        summary = {
            'plates': [],
            'containers': [], 
            'faces': []
        }
        
        for detection in detections:
            if detection.detection_type == 'plate' and detection.text:
                summary['plates'].append(detection.text)
            elif detection.detection_type == 'container' and detection.text:
                summary['containers'].append(detection.text)
            elif detection.detection_type == 'face' and detection.text:
                summary['faces'].append(detection.text)
        
        return summary
    
    def export_results(self, detections: List[DetectionResult], 
                      format_type: str = "json") -> str:
        """
        Export kết quả ra nhiều format
        
        Args:
            detections: Danh sách detection results
            format_type: "json", "csv", "txt"
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
        
