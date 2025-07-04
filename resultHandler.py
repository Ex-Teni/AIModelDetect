import os
import cv2
import base64
import json
import pandas as pd
import requests
import time
from typing import Dict, Optional, Tuple
import asyncio
import websockets
from datetime import datetime

class APITester:
    def __init__(self, images_folder: str = "images/", 
                 real_results_csv: str = "RealResult.csv",
                 compare_results_csv: str = "CompareResult.csv",
                 api_url: str = "http://localhost:8000/process-batch"):
        """
        Khởi tạo API Tester
        
        Args:
            images_folder: Thư mục chứa ảnh test
            real_results_csv: File CSV chứa kết quả thực tế
            compare_results_csv: File CSV lưu kết quả so sánh
            api_url: URL của WebSocket API
        """
        self.images_folder = images_folder
        self.real_results_csv = real_results_csv
        self.compare_results_csv = compare_results_csv
        self.api_url = api_url
        
        # Tạo folder images nếu chưa có
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
            print(f"Created {self.images_folder} folder. Please add test images.")
    
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Chuyển đổi ảnh thành base64"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def call_api(self, image_base64: str) -> Tuple[Optional[Dict], float]:
        """
        Gọi API qua WebSocket và trả về kết quả + thời gian xử lý
        
        Returns:
            Tuple[response_data, processing_time_seconds]
        """
        start_time = time.time()
        try:
            response = requests.post(
                self.api_url,
                json={"images": [image_base64]},
                timeout=90
            )
            processing_time = time.time() - start_time

            if response.status_code != 200:
                print(f"API returned error status: {response.status_code}")
                print("Response:", response.text)
                return None, processing_time
    
            result = response.json()
            if not result.get("success"):
                print(f"API error: {result.get('error')}")
                print("Full response:", result)
                return None, processing_time
    
            # Lấy kết quả đầu tiên (vì gửi 1 ảnh/lần)
            results = result.get("results", [])
            if not results or not results[0].get("success"):
                print("[ERROR] No valid result from API or success=False in result")
                print("Result content:", result)
                return None, processing_time

            return results[0], processing_time

        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            return None, time.time() - start_time
    
    def extract_detection_results(self, api_response: Dict) -> Tuple[str, str, str, float]:
        """
        Trích xuất kết quả plate và container từ API response
        
        Returns:
            Tuple[plate_number, container_code]
        """
        metadata = api_response.get("metadata", {})
        print(f"[TESTER] Metadata: {metadata}")
        detections = api_response.get("detections", [])
        print(f"[TESTER] Raw detections: {detections}")
    
        plate = metadata.get("plate", "None")
        container = metadata.get("container", "None")
        face_name = metadata.get("face", "None")
    
        face_conf = 0.0
        for det in detections:
            if det.get("type") == "face" and det.get("text") == face_name:
                face_conf = float(det.get("confidence", 0.0))
                break
            
        return plate, container, face_name, face_conf
    
    def normalize_text(self, text: Optional[str]) -> str:
        """Chuẩn hóa text để so sánh"""
        if text is None:
            return ""
        return str(text).upper().strip().replace(" ", "").replace("-", "")
    
    def calculate_match(self, actual: Optional[str], predicted: Optional[str]) -> int:
        """
        So sánh kết quả thực tế và dự đoán
        
        Returns:
            1 nếu khớp, 0 nếu không khớp
        """
        actual_norm = self.normalize_text(actual)
        predicted_norm = self.normalize_text(predicted)
        
        if not actual_norm and not predicted_norm:
            return 1  # Cả hai đều rỗng
        elif not actual_norm or not predicted_norm:
            return 0  # Một trong hai rỗng
        else:
            return 1 if actual_norm == predicted_norm else 0
    
    def calculate_accuracy_percentage(self, actual: Optional[str], predicted: Optional[str]) -> float:
        """
        Tính phần trăm chính xác giữa actual và predicted
        
        Returns:
            Phần trăm chính xác (0-100)
        """
        actual_norm = self.normalize_text(actual)
        predicted_norm = self.normalize_text(predicted)
        
        if not actual_norm and not predicted_norm:
            return 100.0  # Cả hai đều rỗng
        elif not actual_norm or not predicted_norm:
            return 0.0  # Một trong hai rỗng
        
        # Tính Levenshtein distance
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(actual_norm), len(predicted_norm))
        if max_len == 0:
            return 100.0
        
        distance = levenshtein_distance(actual_norm, predicted_norm)
        similarity = 1 - (distance / max_len)
        return similarity * 100
    
    async def test_single_image(self, image_name: str, actual_plate: str, actual_container: str, actual_face: str) -> Dict:
        """
        Test một ảnh và trả về kết quả so sánh
        """
        # Nếu thiếu đuôi mở rộng, thử tìm trong thư mục
        possible_exts = ["", ".jpg", ".jpeg", ".png"]
        image_path = None
        for ext in possible_exts:
            candidate = os.path.join(self.images_folder, image_name + ext) if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')) else os.path.join(self.images_folder, image_name)
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        if image_path is None:
            return {
                'Name': image_name,
                'Face_Name': actual_face,
                'Face_Name_AI': 'FILE_NOT_FOUND',
                'Face_Match': 0,
                'Face_Confidence': 0,
                'Plate_Number': actual_plate,
                'Plate_Number_AI': 'FILE_NOT_FOUND',
                'Plate_Accuracy': 0.0,
                'Plate_Match': 0,
                'Container_Code': actual_container,
                'Container_Code_AI': 'FILE_NOT_FOUND',
                'Container_Accuracy': 0.0,
                'Container_Match': 0,
                'Time': 0.0
            }
        
        # Encode ảnh
        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            return {
                'Name': image_name,
                'Face_Name': actual_face,
                'Face_Name_AI': 'ENCODE_ERROR',
                'Face_Match': 0,
                'Face_Confidence': 0,
                'Plate_Number': actual_plate,
                'Plate_Number_AI': 'ENCODE_ERROR',
                'Plate_Match': 0,
                'Plate_Accuracy': 0.0,
                'Container_Code': actual_container,
                'Container_Code_AI': 'ENCODE_ERROR',
                'Container_Match': 0,
                'Container_Accuracy': 0.0,
                'Time': 0.0
            }
        
        # Gọi API
        api_result, processing_time = self.call_api(image_base64)
        
        if api_result is None:
            return {
                'Name': image_name,
                'Face_Name': actual_face,
                'Face_Name_AI': 'API_ERROR',
                'Face_Match': 0,
                'Face_Confidence': 0,
                'Plate_Number': actual_plate,
                'Plate_Number_AI': 'API_ERROR',
                'Plate_Accuracy': 0.0,
                'Plate_Match': 0,
                'Container_Code': actual_container,
                'Container_Code_AI': 'API_ERROR',
                'Container_Accuracy': 0.0,
                'Container_Match': 0,
                'Time': round(processing_time, 3)
            }
        
        # Trích xuất kết quả
        predicted_plate, predicted_container, predicted_face, face_confidence = self.extract_detection_results(api_result)
        
        # Tính toán kết quả
        plate_match = self.calculate_match(actual_plate, predicted_plate)
        plate_accuracy = self.calculate_accuracy_percentage(actual_plate, predicted_plate)

        container_match = self.calculate_match(actual_container, predicted_container)
        container_accuracy = self.calculate_accuracy_percentage(actual_container, predicted_container)

        face_match = 1 if self.normalize_text(actual_face) == self.normalize_text(predicted_face) else 0
        
        return {
            'Name': image_name,
            'Face_Name': actual_face,
            'Face_Name_AI': predicted_face or 'None',
            'Face_Match': face_match,
            'Face_Confidence': round(face_confidence *100, 2) if face_confidence else 0.0,
            'Plate_Number': actual_plate,
            'Plate_Number_AI': predicted_plate or 'None',
            'Plate_Match': plate_match,
            'Plate_Accuracy': round(plate_accuracy, 2),
            'Container_Code': actual_container,
            'Container_Code_AI': predicted_container or 'None',
            'Container_Match': container_match,
            'Container_Accuracy': round(container_accuracy, 2),
            'Time': round(processing_time, 3)
        }
    
    async def run_test(self):
        """Chạy test cho tất cả ảnh trong CSV"""
        print("Starting API testing...")
        print(f"Images folder: {self.images_folder}")
        print(f"Real results CSV: {self.real_results_csv}")
        print(f"API URL: {self.api_url}")
        print("-" * 50)
        
        # Đọc dữ liệu thực tế
        try:
            real_data = pd.read_csv(self.real_results_csv)
        except Exception as e:
            print(f"Error reading {self.real_results_csv}: {e}")
            return
        
        results = []
        total_images = len(real_data)
        
        for index, row in real_data.iterrows():
            image_name = row['Name']
            actual_plate = str(row['Plate_Number']) if pd.notna(row['Plate_Number']) else ''
            actual_container = str(row['Container_Code']) if pd.notna(row['Container_Code']) else ''
            actual_face = str(row['Face_Name']) if pd.notna(row.get('Face_Name')) else ''
            
            print(f"Testing {index + 1}/{total_images}: {image_name}") # type: ignore
            
            result = await self.test_single_image(image_name, actual_plate, actual_container, actual_face)
            results.append(result)
            
            # In kết quả
            print(f"  Plate: {actual_plate} -> {result['Plate_Number_AI']} " f"(Accuracy: {result['Plate_Accuracy']}%, Match: {result['Plate_Match']})")
            print(f"  Container: {actual_container} -> {result['Container_Code_AI']} " f"(Accuracy: {result['Container_Accuracy']}%, Match: {result['Container_Match']})")
            print(f"  Face: {result['Face_Name']} -> {result['Face_Name_AI']} " f"(Confidence: {result['Face_Confidence']}%, Match: {result['Face_Match']})")


            print(f"  Time: {result['Time']}s")
            print()
        
        # Lưu kết quả
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.compare_results_csv, index=False)
        
        # Tính thống kê tổng quan
        total_face_matches = results_df['Face_Match'].sum()
        total_plate_matches = results_df['Plate_Match'].sum()
        total_container_matches = results_df['Container_Match'].sum()
        total_tests = len(results_df)
        
        avg_face_confidence = results_df['Face_Confidence'].mean()
        avg_plate_accuracy = results_df['Plate_Accuracy'].mean()
        avg_container_accuracy = results_df['Container_Accuracy'].mean()
        avg_processing_time = results_df['Time'].mean()
        
        print("=" * 50)
        print("TESTING RESULTS SUMMARY")
        print("=" * 50)
        print(f"Total images tested: {total_tests}")
        print(f"Face recognition:")
        print(f"  - Exact matches: {total_face_matches}/{total_tests} ({total_face_matches/total_tests*100:.1f}%)")
        print(f"  - Average confidence: {avg_face_confidence:.2f}%")
        print(f"Plate detection:")
        print(f"  - Exact matches: {total_plate_matches}/{total_tests} ({total_plate_matches/total_tests*100:.1f}%)")
        print(f"  - Average accuracy: {avg_plate_accuracy:.1f}%")
        print(f"Container detection:")
        print(f"  - Exact matches: {total_container_matches}/{total_tests} ({total_container_matches/total_tests*100:.1f}%)")
        print(f"  - Average accuracy: {avg_container_accuracy:.1f}%")
        print(f"Average processing time: {avg_processing_time:.3f}s")
        print(f"Results saved to: {self.compare_results_csv}")

def main():
    """Hàm main để chạy test"""
    # Khởi tạo tester
    tester = APITester(
        images_folder="images",
        real_results_csv="realResult.csv", 
        compare_results_csv="CompareResult.csv",
        api_url="http://localhost:8000/process-batch"
    )
    
    # Chạy test
    asyncio.run(tester.run_test())

if __name__ == "__main__":
    main()