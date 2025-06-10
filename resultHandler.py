import csv
import json
import base64
import asyncio
import websockets
import time
import os
from typing import Dict, Optional

class Comparator:
    def __init__(self):
        pass
        
    def load_labels(self, csv_path: str) -> Dict[str, dict]:
        """Đọc kết quả thực tế từ file CSV"""
        labels = {}
        try:
            with open(csv_path, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                # Làm sạch tên cột
                if reader.fieldnames:
                    reader.fieldnames = [field.strip() for field in reader.fieldnames]
                    
                for row in reader:
                    # Làm sạch key trong từng dòng
                    row = {k.strip(): v.strip() if v else "" for k, v in row.items()}
                    filename = row.get("Name", "").strip()
                    
                    if filename:
                        labels[filename] = {
                            'plate': row.get("Plate_Number", "").strip(),
                            'container': row.get("Container_Code", "").strip()
                        }
                        
            print(f"[INFO] Loaded {len(labels)} records from {csv_path}")
            return labels
            
        except Exception as e:
            print(f"[ERROR] Read file error {csv_path}: {e}")
            raise

    def compare_exact(self, real_result: str, ai_result: str) -> int:
        """So sánh chính xác, trả về 1 nếu đúng, 0 nếu sai"""
        if not real_result and not ai_result:
            return 1
        if not real_result or not ai_result:
            return 0
        return 1 if real_result.lower() == ai_result.lower() else 0

    def encode_image(self, image_path: str) -> str:
        """Đọc và encode ảnh thành base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    async def process_single_image(self, ws, image_path: str, filename: str) -> tuple:
        """Xử lý một ảnh và nhận kết quả từ API"""
        try:
            start_time = time.time()
            
            # Gửi ảnh tới API
            image_b64 = self.encode_image(image_path)
            message = {
                "image": image_b64,
                "filename": filename
            }
            
            await ws.send(json.dumps(message))
            
            # Nhận kết quả
            response = await ws.recv()
            data = json.loads(response)
            
            end_time = time.time()
            detection_time = round((end_time - start_time) * 1000, 2)  # ms
            
            if not data.get("success", False):
                print(f"[ERROR] API error {filename}: {data.get('error', 'Unknown error')}")
                return None, detection_time, 0.0
                
            metadata = data.get("metadata", {})
            
            # Lấy confidence nếu có trong detections
            detections = data.get("detections", [])
            avg_confidence = 0.0
            if detections:
                confidences = [det.get("confidence", 0.0) for det in detections if det.get("confidence")]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                avg_confidence = round(avg_confidence * 100, 2)  # chuyển thành %
                
            return metadata, detection_time, avg_confidence
            
        except Exception as e:
            print(f"[ERROR] Images error {filename}: {e}")
            return None, 0, 0.0

    async def process_all_images(self, uri: str, labels_csv: str, output_csv: str, images_dir: str = "images"):
        """Xử lý tất cả ảnh và so sánh kết quả"""
        labels = self.load_labels(labels_csv)
        results = []
        
        try:
            async with websockets.connect(uri) as ws:
                print("[CONNECTED] Connected to API")
                
                for filename, expected in labels.items():
                    # Tìm file ảnh
                    image_path = None
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        potential_path = os.path.join(images_dir, f"{filename}{ext}")
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break
                    
                    if not image_path:
                        print(f"[WARNING] Không tìm thấy ảnh cho {filename}")
                        # Thêm record với kết quả fail
                        results.append({
                            "Name": filename,
                            "Plate_Expected": expected["plate"],
                            "Plate_AI": "",
                            "Plate_Match": 0,
                            "Container_Expected": expected["container"],
                            "Container_AI": "",
                            "Container_Match": 0,
                            "Detection_Time_ms": 0,
                            "Confidence_Percent": 0.0,
                            "Status": "Image_Not_Found"
                        })
                        continue
                    
                    # Xử lý ảnh
                    metadata, detection_time, confidence = await self.process_single_image(ws, image_path, filename)
                    
                    if metadata is None:
                        # API lỗi
                        results.append({
                            "Name": filename,
                            "Plate_Expected": expected["plate"],
                            "Plate_AI": "",
                            "Plate_Match": 0,
                            "Container_Expected": expected["container"],
                            "Container_AI": "",
                            "Container_Match": 0,
                            "Detection_Time_ms": detection_time,
                            "Confidence_Percent": confidence,
                            "Status": "API_Error"
                        })
                        continue
                    
                    # Lấy kết quả AI
                    plate_ai = metadata.get("plate", "").strip()
                    container_ai = metadata.get("container", "").strip()
                    
                    # So sánh kết quả (0 hoặc 1)
                    plate_match = self.compare_exact(expected['plate'], plate_ai)
                    container_match = self.compare_exact(expected['container'], container_ai)
                    
                    # Lưu kết quả
                    result = {
                        "Name": filename,
                        "Plate_Expected": expected["plate"],
                        "Plate_AI": plate_ai,
                        "Plate_Match": plate_match,
                        "Container_Expected": expected["container"],
                        "Container_AI": container_ai,
                        "Container_Match": container_match,
                        "Detection_Time_ms": detection_time,
                        "Confidence_Percent": confidence,
                        "Status": "Success"
                    }
                    
                    results.append(result)
                    
                    print(f"[RESULT] [{filename}] Plate: {plate_match} | Container: {container_match} | "
                          f"Time: {detection_time}ms | Confidence: {confidence}%")
                    
                    # Delay nhỏ
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"[ERROR] Error connect to API: {e}")
            raise
        
        # Lưu kết quả
        self.save_results(results, output_csv)
        self.print_summary(results)

    def save_results(self, results: list, output_csv: str):
        """Lưu kết quả ra file CSV"""
        if not results:
            print("[WARNING] No result to save")
            return
            
        fieldnames = [
            "Name", 
            "Plate_Expected", "Plate_AI", "Plate_Match",
            "Container_Expected", "Container_AI", "Container_Match",
            "Detection_Time_ms", "Confidence_Percent", "Status"
        ]
        
        try:
            with open(output_csv, mode='w', newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"[SUCCESS] Saved {len(results)} result to {output_csv}")
        except Exception as e:
            print(f"[ERROR] Error saved file {output_csv}: {e}")

    def print_summary(self, results: list):
        """In tóm tắt kết quả"""
        if not results:
            return
            
        successful_results = [r for r in results if r["Status"] == "Success"]
        total_success = len(successful_results)
        
        if total_success == 0:
            print("[WARNING] None successed!")
            return
        
        # Tính tỷ lệ đúng
        plate_correct = sum(r["Plate_Match"] for r in successful_results)
        container_correct = sum(r["Container_Match"] for r in successful_results)
        
        # Thời gian detect trung bình
        avg_time = sum(r["Detection_Time_ms"] for r in successful_results) / total_success
        
        # Confidence trung bình
        avg_confidence = sum(r["Confidence_Percent"] for r in successful_results) / total_success
        
        print("\n[INFO] === TÓM TẮT KẾT QUẢ ===")
        print(f"[INFO] Success compare: {total_success}/{len(results)}")
        print(f"[INFO] Plate number - Match: {plate_correct}/{total_success} ({plate_correct/total_success*100:.1f}%)")
        print(f"[INFO] Container - Match: {container_correct}/{total_success} ({container_correct/total_success*100:.1f}%)")
        print(f"[INFO] Average detect time: {avg_time:.1f}ms")
        print(f"[INFO] Confidence: {avg_confidence:.1f}%")


async def main():
    """Hàm main"""
    # Cấu hình - Chỉ cần thay đổi ở đây
    CONFIG = {
        "api_url": "ws://localhost:8000/ws/combined-detection",
        "input_csv": "RealResult.csv",          # File chứa kết quả thực tế
        "output_csv": "ComparisonResults.csv",   # File kết quả so sánh
        "images_folder": "images"                # Thư mục chứa ảnh
    }
    
    # Kiểm tra file và thư mục tồn tại
    if not os.path.exists(CONFIG["input_csv"]):
        print(f"[ERROR] File {CONFIG['input_csv']} not exists!")
        return
        
    if not os.path.exists(CONFIG["images_folder"]):
        print(f"[ERROR] Images folder {CONFIG['images_folder']} not exists!")
        return
    
    print("[MESSAGE] Comparring...")
    
    # Chạy so sánh
    comparator = Comparator()
    try:
        await comparator.process_all_images(
            CONFIG["api_url"], 
            CONFIG["input_csv"], 
            CONFIG["output_csv"],
            CONFIG["images_folder"]
        )
        print("[SUCCESS] Comparring success!")
        
    except KeyboardInterrupt:
        print("\n[WARNING] STOPPED!")
    except Exception as e:
        print(f"[ERROR] Error in comparing: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("[MESSAGE] Compare system")
    print("=" * 50)
    asyncio.run(main())