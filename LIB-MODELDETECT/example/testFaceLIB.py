import os
import cv2
import time
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

# Import đúng class MultiDetect_Core từ lib
from lib import MultiDetect_Core

class FaceTester:
    def __init__(self, faces_folder: str, real_results_csv: str, compare_results_csv: str, device: str = "auto"):
        self.faces_folder = faces_folder
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
        if not actual or actual.strip().upper() == "NONE":
            return -1
        actual_norm = self.normalize_text(actual)
        predicted_norm = self.normalize_text(predicted)
        if not predicted_norm:
            return 0
        return 1 if actual_norm == predicted_norm else 0

    def calculate_accuracy_percentage(self, actual: Optional[str], predicted: Optional[str]) -> float:
        if not actual or actual.strip().upper() == "NONE":
            return -1.0
        actual_norm = self.normalize_text(actual)
        predicted_norm = self.normalize_text(predicted)
        if not actual_norm and not predicted_norm:
            return 0.0
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

    def test_single_image(self, image_name: str, actual_face: str) -> Dict:
        # tìm file ảnh trong folder
        possible_exts = ["", ".jpg", ".jpeg", ".png"]
        image_path = None
        for ext in possible_exts:
            candidate = os.path.join(self.faces_folder, image_name + ext) if not image_name.lower().endswith(
                ('.jpg', '.png', '.jpeg')) else os.path.join(self.faces_folder, image_name)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            return {
                'Name': image_name,
                'Face_Name': actual_face,
                'Face_Name_AI': 'FILE_NOT_FOUND',
                'Face_Match': -1,
                'Face_Confidence': -1.0,
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
                'Face_Name': actual_face,
                'Face_Name_AI': 'LOAD_ERROR',
                'Face_Match': -1,
                'Face_Confidence': -1.0,
                'Time': 0.0
            }

        # detect
        start_time = time.time()
        detections = self.detector.detect_faces_only(image)
        elapsed = time.time() - start_time

        # lấy text tốt nhất
        predicted_face = None
        best_score = -1.0
        recog_score = 0
        for det in detections:
            if getattr(det, "detection_type", None) == "face":
                text = getattr(det, "text", None)
                score = getattr(det, "recognition_confidence", getattr(det, "confidence", 0.0))
                try:
                    score = float(score)
                except Exception:
                    score = 0.0
                if text and score > best_score:
                    predicted_face, best_score = str(text), score
                    recog_score = score

        # chấm điểm
        Face_Match = self.calculate_match(actual_face, predicted_face)

        return {
            'Name': image_name,
            'Face_Name': actual_face,
            'Face_Name_AI': predicted_face or 'None',
            'Face_Match': Face_Match,
            # dùng % recognition_confidence thay vì levenshtein accuracy
            'Face_Confidence': round(recog_score * 100, 2),
            'Time': round(elapsed, 3)
        }

    def run_test(self):
        print("Starting face Library testing...")
        print(f"Faces folder: {self.faces_folder}")
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
            actual_face = str(row['Face_Name']) if pd.notna(row['Face_Name']) else 'None'

            print(f"Testing {index + 1}/{total_images}: {image_name}")
            result = self.test_single_image(image_name, actual_face)
            results.append(result)

            print(f"  face: {actual_face} -> {result['Face_Name_AI']} "
                  f"(Accuracy: {result['Face_Confidence']}%, Match: {result['Face_Match']})")
            print(f"  Time: {result['Time']}s\n")

        results_df = pd.DataFrame(results)
        Path(self.compare_results_csv).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.compare_results_csv, index=False)

        # tính toán thống kê
        valid_face_results = results_df[results_df['Face_Match'] != -1]
        total_Face_Matches = int(valid_face_results['Face_Match'].sum()) if len(valid_face_results) > 0 else 0
        avg_Face_Confidence = float(valid_face_results['Face_Confidence'].mean()) if len(valid_face_results) > 0 else 0.0
        avg_processing_time = float(results_df['Time'].mean()) if len(results_df) > 0 else 0.0

        print("=" * 60)
        print("face DETECTION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total images tested: {len(results_df)}\n")

        print(f"  - Images with ground truth: {len(valid_face_results)}")
        if len(valid_face_results) > 0:
            print(f"  - Exact matches: {total_Face_Matches}/{len(valid_face_results)} "
                  f"({(total_Face_Matches / len(valid_face_results) * 100):.1f}%)")
            print(f"  - Average accuracy: {avg_Face_Confidence:.1f}%")
        else:
            print("  - No valid face data for evaluation")

        print(f"\nPERFORMANCE:")
        print(f"  - Average processing time: {avg_processing_time:.3f}s\n")
        print(f"Results saved to: {self.compare_results_csv}")


def main():
    root_path = Path(__file__).parent.parent

    face_folder = str(root_path / "example" / "images" / "face")
    real_csv_face = str(root_path / "example" / "output" / "face" / "RealResult.csv")   
    compare_csv_face = str(root_path / "example" / "output" / "face" / "CompareResult.csv")  

    tester = FaceTester(face_folder, real_csv_face, compare_csv_face, device="auto")
    tester.run_test()


if __name__ == "__main__":
    main()