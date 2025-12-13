import os
import cv2
import time
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

# Import đúng class MultiDetect_Core từ lib
from lib import MultiDetect_Core

class PlateTester:
    def __init__(self, plates_folder: str, real_results_csv: str, compare_results_csv: str, device: str = "auto"):
        self.plates_folder = plates_folder
        self.real_results_csv = real_results_csv
        self.compare_results_csv = compare_results_csv
        self.device = device

        # Khởi tạo thư viện detect
        self.detector = MultiDetect_Core(device=device)

    def normalize_text(self, text: Optional[str]) -> str:
        if text is None or text == "None" or text == "":
            return ""
        return str(text).upper().strip().replace(" ", "").replace("-", "")

    def calculate_match(self, actual: Optional[str], predicted: Optional[str]) -> int:
        "So sánh độ trùng hợp của LIB - Dữ liệu thực tế"
        # TH1: Không có ground truth
        if not actual or actual.strip().upper() == "NONE":
            if predicted and predicted.strip() != "" and predicted.strip().upper() != "NONE":
                return 0  # False Positive → tính là sai
            return -1     # None vs None → bỏ qua
        # TH2: Có ground truth
        actual_norm = self.normalize_text(actual)
        predicted_norm = self.normalize_text(predicted)
        if not predicted_norm:
            return 0
        return 1 if actual_norm == predicted_norm else 0

    def calculate_accuracy_percentage(self, actual: Optional[str], predicted: Optional[str]) -> float:
        # TH1: Không có ground truth
        if not actual or actual.strip().upper() == "NONE":
            if predicted and predicted.strip() != "" and predicted.strip().upper() != "NONE":
                return 0.0   # False Positive → phạt điểm
            return -1.0      # None vs None → bỏ qua
        # TH2: Có ground truth
        actual_norm = self.normalize_text(actual)
        predicted_norm = self.normalize_text(predicted)
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

    def test_single_image(self, image_name: str, actual_plate: str) -> Dict:
        # tìm file ảnh trong folder
        possible_exts = ["", ".jpg", ".jpeg", ".png"]
        image_path = None
        for ext in possible_exts:
            candidate = os.path.join(self.plates_folder, image_name + ext) if not image_name.lower().endswith(
                ('.jpg', '.png', '.jpeg')) else os.path.join(self.plates_folder, image_name)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            return {
                'Name': image_name,
                'Plate_Number': actual_plate,
                'Plate_Number_AI': 'FILE_NOT_FOUND',
                'Plate_Match': -1,
                'Plate_Accuracy': -1.0,
                'Failed_Reason': failed_reason or 'FILE_NOT_FOUND',
                'Time': 0.0
            }

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Can't read image")
        except Exception as e:
            print(f"Cannot read {image_name}: {e}")
            return {
                'Name': image_name,
                'Plate_Number': actual_plate,
                'Plate_Number_AI': 'LOAD_ERROR',
                'Plate_Match': -1,
                'Plate_Accuracy': -1.0,
                'Failed_Reason': failed_reason or 'LOAD_ERROR',
                'Time': 0.0
            }

        # detect
        start_time = time.time()
        detections = self.detector.detect_plates_only(image)
        elapsed = time.time() - start_time

        # lấy text tốt nhất
        predicted_plate = None
        best_score = -1.0
        failed_reason = ""
        for det in detections:
            if getattr(det, "detection_type", None) == "plate":
                text = getattr(det, "text", None)
                score = getattr(det, "ocr_confidence", getattr(det, "confidence", 0.0))
                reason = getattr(det, "failed_reason", None) 
                try:
                    score = float(score)
                except Exception:
                    score = 0.0
                if text and score > best_score:
                    predicted_plate, best_score, failed_reason = str(text), score, reason

        # chấm điểm
        plate_match = self.calculate_match(actual_plate, predicted_plate)
        plate_accuracy = self.calculate_accuracy_percentage(actual_plate, predicted_plate)

        return {
            'Name': image_name,
            'Plate_Number': actual_plate,
            'Plate_Number_AI': predicted_plate or 'None',
            'Plate_Match': plate_match,
            'Plate_Accuracy': round(plate_accuracy, 2) if plate_accuracy != -1 else -1,
            'Failed_Reason': failed_reason,
            'Time': round(elapsed, 3)
        }

    def run_test(self):
        print("Starting Plate Library testing...")
        print(f"Plates folder: {self.plates_folder}")
        print(f"Real results CSV: {self.real_results_csv}")
        print("-" * 50)

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

            print(f"Testing {index + 1}/{total_images}: {image_name}")
            result = self.test_single_image(image_name, actual_plate)
            results.append(result)

            print(f"  Plate: {actual_plate} -> {result['Plate_Number_AI']} "
                  f"(Accuracy: {result['Plate_Accuracy']}%, Match: {result['Plate_Match']})")
            print(f"  Time: {result['Time']}s  | Failed_Reason: {result['Failed_Reason']}\n")

        results_df = pd.DataFrame(results)
        Path(self.compare_results_csv).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.compare_results_csv, index=False)

        # tính toán thống kê
        valid_plate_results = results_df[results_df['Plate_Match'] != -1]
        total_plate_matches = int(valid_plate_results['Plate_Match'].sum()) if len(valid_plate_results) > 0 else 0
        avg_plate_accuracy = float(valid_plate_results['Plate_Accuracy'].mean()) if len(valid_plate_results) > 0 else 0.0
        avg_processing_time = float(results_df['Time'].mean()) if len(results_df) > 0 else 0.0

        print("=" * 60)
        print("PLATE DETECTION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total images tested: {len(results_df)}\n")

        print(f"  - Images with ground truth: {len(valid_plate_results)}")
        if len(valid_plate_results) > 0:
            print(f"  - Exact matches: {total_plate_matches}/{len(valid_plate_results)} "
                  f"({(total_plate_matches / len(valid_plate_results) * 100):.1f}%)")
            print(f"  - Average accuracy: {avg_plate_accuracy:.1f}%")
        else:
            print("  - No valid plate data for evaluation")

        print(f"\nPERFORMANCE:")
        print(f"  - Average processing time: {avg_processing_time:.3f}s\n")
        print(f"Results saved to: {self.compare_results_csv}")


def main():
    root_path = Path(__file__).parent.parent

    plate_folder = str(root_path / "example" / "images" / "plate")
    real_csv_plate = str(root_path / "example" / "output" / "plate" / "RealResult.csv")   
    compare_csv_plate = str(root_path / "example" / "output" / "plate" / "CompareResult.csv")  

    tester = PlateTester(plate_folder, real_csv_plate, compare_csv_plate, device="auto")
    tester.run_test()


if __name__ == "__main__":
    main()
