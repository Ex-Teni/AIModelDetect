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
        self.images_folder = images_folder
        self.real_results_csv = real_results_csv
        self.compare_results_csv = compare_results_csv
        self.api_url = api_url

        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
            print(f"Created {self.images_folder} folder. Please add test images.")

    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def call_api(self, image_base64: str) -> Tuple[Optional[Dict], float]:
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
        if text is None or text == "None" or text == "":
            return ""
        return str(text).upper().strip().replace(" ", "").replace("-", "")

    def is_none_value(self, value: Optional[str]) -> bool:
        """Kiểm tra xem giá trị có phải là None/rỗng không"""
        if value is None:
            return True
        if isinstance(value, str) and (value.strip() == "" or value.strip().upper() == "NONE"):
            return True
        return False

    def calculate_match(self, actual: Optional[str], predicted: Optional[str]) -> int:
        """
        Tính exact match
        - Trả về 1 nếu khớp chính xác
        - Trả về 0 nếu không khớp
        - Trả về -1 nếu actual là None (không tính trong thống kê)
        """
        # Nếu actual là None, không tính vào thống kê
        if self.is_none_value(actual):
            return -1
            
        actual_norm = self.normalize_text(actual)
        predicted_norm = self.normalize_text(predicted)
        
        # Nếu actual có giá trị nhưng predicted là None
        if not predicted_norm:
            return 0
            
        return 1 if actual_norm == predicted_norm else 0

    def calculate_accuracy_percentage(self, actual: Optional[str], predicted: Optional[str]) -> float:
        """
        Tính độ chính xác dựa trên Levenshtein distance
        - Trả về -1 nếu actual là None (không tính trong thống kê)
        - Trả về 0 nếu cả actual và predicted đều None
        - Trả về % accuracy cho các trường hợp khác
        """
        # Nếu actual là None, không tính vào thống kê
        if self.is_none_value(actual):
            return -1.0
            
        actual_norm = self.normalize_text(actual)
        predicted_norm = self.normalize_text(predicted)
        
        # Nếu cả actual và predicted đều None (sau khi normalize)
        if not actual_norm and not predicted_norm:
            return 0.0
            
        # Nếu actual có giá trị nhưng predicted là None
        if not predicted_norm:
            return 0.0

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
            return 0.0

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
                'Face_Match': -1,
                'Face_Confidence': 0,
                'Plate_Number': actual_plate,
                'Plate_Number_AI': 'FILE_NOT_FOUND',
                'Plate_Accuracy': -1.0,
                'Plate_Match': -1,
                'Container_Code': actual_container,
                'Container_Code_AI': 'FILE_NOT_FOUND',
                'Container_Accuracy': -1.0,
                'Container_Match': -1,
                'Time': 0.0
            }
        
        # Encode ảnh
        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            return {
                'Name': image_name,
                'Face_Name': actual_face,
                'Face_Name_AI': 'ENCODE_ERROR',
                'Face_Match': -1,
                'Face_Confidence': 0,
                'Plate_Number': actual_plate,
                'Plate_Number_AI': 'ENCODE_ERROR',
                'Plate_Match': -1,
                'Plate_Accuracy': -1.0,
                'Container_Code': actual_container,
                'Container_Code_AI': 'ENCODE_ERROR',
                'Container_Match': -1,
                'Container_Accuracy': -1.0,
                'Time': 0.0
            }
        
        # Gọi API
        api_result, processing_time = self.call_api(image_base64)
        
        if api_result is None:
            return {
                'Name': image_name,
                'Face_Name': actual_face,
                'Face_Name_AI': 'API_ERROR',
                'Face_Match': -1,
                'Face_Confidence': 0,
                'Plate_Number': actual_plate,
                'Plate_Number_AI': 'API_ERROR',
                'Plate_Accuracy': -1.0,
                'Plate_Match': -1,
                'Container_Code': actual_container,
                'Container_Code_AI': 'API_ERROR',
                'Container_Accuracy': -1.0,
                'Container_Match': -1,
                'Time': round(processing_time, 3)
            }
        
        # Trích xuất kết quả
        predicted_plate, predicted_container, predicted_face, face_confidence = self.extract_detection_results(api_result)
        
        # Tính toán kết quả
        plate_match = self.calculate_match(actual_plate, predicted_plate)
        plate_accuracy = self.calculate_accuracy_percentage(actual_plate, predicted_plate)

        container_match = self.calculate_match(actual_container, predicted_container)
        container_accuracy = self.calculate_accuracy_percentage(actual_container, predicted_container)

        face_match = self.calculate_match(actual_face, predicted_face)
        
        return {
            'Name': image_name,
            'Face_Name': actual_face,
            'Face_Name_AI': predicted_face or 'None',
            'Face_Match': face_match,
            'Face_Confidence': round(face_confidence * 100, 2) if face_confidence else 0.0,
            'Plate_Number': actual_plate,
            'Plate_Number_AI': predicted_plate or 'None',
            'Plate_Match': plate_match,
            'Plate_Accuracy': round(plate_accuracy, 2) if plate_accuracy != -1 else -1,
            'Container_Code': actual_container,
            'Container_Code_AI': predicted_container or 'None',
            'Container_Match': container_match,
            'Container_Accuracy': round(container_accuracy, 2) if container_accuracy != -1 else -1,
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
            actual_plate = str(row['Plate_Number']) if pd.notna(row['Plate_Number']) else 'None'
            actual_container = str(row['Container_Code']) if pd.notna(row['Container_Code']) else 'None'
            actual_face = str(row['Face_Name']) if pd.notna(row.get('Face_Name')) else 'None'
            
            print(f"Testing {index + 1}/{total_images}: {image_name}") # type: ignore
            
            result = await self.test_single_image(image_name, actual_plate, actual_container, actual_face)
            results.append(result)
            
            # In kết quả với thông tin chi tiết
            if result['Plate_Match'] == -1:
                print(f"  Plate: {actual_plate} (SKIPPED - No ground truth)")
            else:
                print(f"  Plate: {actual_plate} -> {result['Plate_Number_AI']} "
                      f"(Accuracy: {result['Plate_Accuracy']}%, Match: {result['Plate_Match']})")
            
            if result['Container_Match'] == -1:
                print(f"  Container: {actual_container} (SKIPPED - No ground truth)")
            else:
                print(f"  Container: {actual_container} -> {result['Container_Code_AI']} "
                      f"(Accuracy: {result['Container_Accuracy']}%, Match: {result['Container_Match']})")
            
            if result['Face_Match'] == -1:
                print(f"  Face: {actual_face} (SKIPPED - No ground truth)")
            else:
                print(f"  Face: {result['Face_Name']} -> {result['Face_Name_AI']} "
                      f"(Confidence: {result['Face_Confidence']}%, Match: {result['Face_Match']})")

            print(f"  Time: {result['Time']}s")
            print()
        
        # Lưu kết quả
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.compare_results_csv, index=False)

        # Tính thống kê - chỉ cho các trường hợp có ground truth (không phải -1)
        valid_plate_results = results_df[results_df['Plate_Match'] != -1]
        valid_container_results = results_df[results_df['Container_Match'] != -1]
        valid_face_results = results_df[results_df['Face_Match'] != -1]

        # Tính số lượng matches
        total_plate_matches = valid_plate_results['Plate_Match'].sum()
        total_container_matches = valid_container_results['Container_Match'].sum()
        total_face_matches = valid_face_results['Face_Match'].sum()

        # Tính accuracy trung bình - chỉ cho các trường hợp có ground truth
        avg_plate_accuracy = valid_plate_results['Plate_Accuracy'].mean() if len(valid_plate_results) > 0 else 0
        avg_container_accuracy = valid_container_results['Container_Accuracy'].mean() if len(valid_container_results) > 0 else 0
        avg_face_confidence = valid_face_results['Face_Confidence'].mean() if len(valid_face_results) > 0 else 0

        avg_processing_time = results_df['Time'].mean()
        
        print("=" * 60)
        print("TESTING RESULTS SUMMARY (STRICT EVALUATION)")
        print("=" * 60)
        print(f"Total images tested: {len(results_df)}")
        print()

        print(f"FACE RECOGNITION:")
        print(f"  - Images with ground truth: {len(valid_face_results)}")
        if len(valid_face_results) > 0:
            print(f"  - Exact matches: {total_face_matches}/{len(valid_face_results)} "
                  f"({(total_face_matches/len(valid_face_results)*100):.1f}%)")
            print(f"  - Average confidence: {avg_face_confidence:.2f}%")
        else:
            print(f"  - No valid face data for evaluation")
        print()

        print(f"PLATE DETECTION:")
        print(f"  - Images with ground truth: {len(valid_plate_results)}")
        if len(valid_plate_results) > 0:
            print(f"  - Exact matches: {total_plate_matches}/{len(valid_plate_results)} "
                  f"({(total_plate_matches/len(valid_plate_results)*100):.1f}%)")
            print(f"  - Average accuracy: {avg_plate_accuracy:.1f}%")
        else:
            print(f"  - No valid plate data for evaluation")
        print()

        print(f"CONTAINER DETECTION:")
        print(f"  - Images with ground truth: {len(valid_container_results)}")
        if len(valid_container_results) > 0:
            print(f"  - Exact matches: {total_container_matches}/{len(valid_container_results)} "
                  f"({(total_container_matches/len(valid_container_results)*100):.1f}%)")
            print(f"  - Average accuracy: {avg_container_accuracy:.1f}%")
        else:
            print(f"  - No valid container data for evaluation")
        print()

        print(f"PERFORMANCE:")
        print(f"  - Average processing time: {avg_processing_time:.3f}s")
        print()
        print(f"Results saved to: {self.compare_results_csv}")
        print("Note: Only images with ground truth data are included in accuracy calculations")

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