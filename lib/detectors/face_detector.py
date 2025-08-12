from typing import List, Optional
import importlib.resources as pkg_resources
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import joblib

from ..results import DetectionResult, FaceResult
from .base_detector import BaseDetector

class FaceDetector(BaseDetector):
    """Detector cho khuôn mặt"""
    
    def _load_model(self):
        """Load các model cần thiết cho face detection và recognition"""
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.face_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # Load classifier/label encoder từ lib.model nếu có
        self.face_classifier = None
        self.label_encoder = None
        try:
            with pkg_resources.path('lib.model', 'face_classifier.joblib') as clf_p:
                with pkg_resources.path('lib.model', 'label_encoder.joblib') as le_p:
                    if clf_p.exists() and le_p.exists():
                        self.face_classifier = joblib.load(clf_p)
                        self.label_encoder = joblib.load(le_p)
                        print("[INFO] Face classifier & label encoder loaded.")
        except Exception as e:
            print(f"[WARNING] Face classifier assets not found or failed: {e}")

    @torch.no_grad()
    def _embed(self, face_tensor: torch.Tensor) -> np.ndarray:
        return self.facenet(face_tensor.to(self.device)).cpu().numpy()

    def detect(self, image: np.ndarray) -> List[DetectionResult]:  # type: ignore
        """
        Phát hiện và nhận dạng khuôn mặt trong ảnh
        Args:
            image: Ảnh đầu vào (BGR format)
        Returns:
            List[DetectionResult]: Danh sách khuôn mặt phát hiện được
        """
        results: List[DetectionResult] = []
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn.detect(img_rgb)  # type: ignore
            if boxes is None or len(boxes) == 0:
                return results
            for i, box in enumerate(boxes):
                det_conf = float(probs[i]) if probs is not None else 0.0
                if det_conf < 0.1:
                    continue
                x1, y1, x2, y2 = map(int, box)
                h, w = img_rgb.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                face_crop = img_rgb[y1:y2, x1:x2]
                person_name = "Unknown"
                rec_conf = 0.0
                try:
                    face_tensor = self.face_transform(Image.fromarray(face_crop)).unsqueeze(0)  # type: ignore
                    emb = self._embed(face_tensor)
                    if self.face_classifier is not None and self.label_encoder is not None:
                        proba = self.face_classifier.predict_proba(emb)  # type: ignore
                        best_idx = int(np.argmax(proba))
                        rec_conf = float(proba[best_idx])
                        if rec_conf >= 0.15:
                            person_name = str(self.label_encoder.inverse_transform([best_idx]))  # type: ignore
                except Exception as e:
                    print(f"[ERROR] Face recognition error: {e}")
                results.append(FaceResult(
                    detection_type='face',
                    bbox=[x1, y1, x2, y2],
                    confidence=float(max(det_conf, rec_conf)),
                    text=person_name,
                    detection_confidence=det_conf,
                    recognition_confidence=rec_conf if person_name != "Unknown" else 0.0,
                    person_name=person_name
                ))
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
        return results
