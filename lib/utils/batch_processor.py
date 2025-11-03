import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Callable
import cv2
import json

class BatchProcessor:
    """Utility cho xử lý batch"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process_folder(self, folder_path: str, 
                      detection_func: Callable,
                      output_format: str = "json") -> Dict[str, Any]:
        """Xử lý toàn bộ folder ảnh"""
        folder = Path(folder_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        results = {}
        
        def process_single_image(image_path):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    return str(image_path), None
                
                detections = detection_func(image)
                return str(image_path), detections
            except Exception as e:
                print(f"[ERROR] Processing {image_path}: {e}")
                return str(image_path), None
        
        # Xử lý song song
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_single_image, img_path) 
                      for img_path in image_files]
            
            for future in concurrent.futures.as_completed(futures):
                img_path, detections = future.result()
                results[img_path] = detections
        
        # Format output
        if output_format == "json":
            return self._format_json_output(results)
        elif output_format == "text":
            return self._format_text_output(results)
        elif output_format == "csv":
            return self._format_csv_output(results)
        
        return results
    
    def _format_json_output(self, results: Dict) -> Dict[str, Any]:
        """Format output thành JSON"""
        formatted = {
            'total_images': len(results),
            'successful_detections': sum(1 for v in results.values() if v is not None),
            'results': {}
        }
        
        for img_path, detections in results.items():
            if detections:
                formatted['results'][img_path] = [d.to_dict() for d in detections]
            else:
                formatted['results'][img_path] = None
        
        return formatted
    
    def _format_text_output(self, results: Dict) -> str:
        """Format output thành text"""
        lines = []
        for img_path, detections in results.items():
            lines.append(f"\n--- {Path(img_path).name} ---")
            if detections:
                for detection in detections:
                    lines.append(detection.to_text_summary())
            else:
                lines.append("No detections")
        
        return "\n".join(lines)