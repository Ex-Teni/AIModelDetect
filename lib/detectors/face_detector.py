import numpy as np
import faiss
import importlib.resources as pkg_resources
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pickle

from ..results import FaceResult
from .base_detector import BaseDetector

class FaceDetector(BaseDetector):
    """FaceDetector sử dụng FaceNet-PyTorch với FAISS Database"""
    
    def __init__(self, 
                device: str = "auto",
                model_name = "vggface2",
                faiss_name = "face_database.faiss",
                labels_name = "face_labels.pkl"):
        
        self._init_device_arg = device
        self._model_name = model_name
        self._faiss_name = faiss_name
        self._labels_name = labels_name

        # chuẩn bị các placeholder
        self.mtcnn = None
        self.resnet = None
        self.sim_threshold = 0.5
        self.similarity_metric = "cosine"

        # Database components
        self.faiss_index = None
        self.labels = None
        self.names = []
        self.class_map = {}
        self.database_loaded = False

        # Performance stats
        self.stats = {
            'total_detections': 0,
            'known_faces': 0,
            'unknown_faces': 0,
            'avg_detection_time': 0.0
        }

        super().__init__(device)

        # Load database if paths provided
        self.load_face_database()

    def _load_model(self):
        """Load FaceNet models (MTCNN + ResNet)"""
        try:
            device = self.device if hasattr(self, "device") else torch.device("cpu")
            
            print(f"[INFO] Loading FaceNet models on: {device}")
            print(f"[INFO] CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[INFO] GPU: {torch.cuda.get_device_name()}")
                print(f"[INFO] CUDA Version: {torch.version.cuda}")

            # Initialize MTCNN for face detection
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                min_face_size=40,
                thresholds=[0.5, 0.6, 0.7],
                factor=0.709,
                post_process=True,
                device=device,
                keep_all=True
            )

            # Initialize InceptionResnetV1 for embeddings
            self.resnet = InceptionResnetV1(pretrained=self._model_name).eval().to(device)

            # Warm up models
            self._warm_up_models()

            print(f"[SUCCESS] FaceNet models loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load FaceNet models: {e}")

    def _warm_up_models(self):
        """Warm up models for faster first inference"""
        try:
            print("[INFO] Warming up models...")
            
            dummy_img = Image.new('RGB', (640, 640), color='white')
            _ = self.mtcnn.detect(dummy_img)
            
            dummy_tensor = torch.randn(1, 3, 160, 160).to(next(self.resnet.parameters()).device)
            with torch.no_grad():
                _ = self.resnet(dummy_tensor)
            
            print("[SUCCESS] Model warm-up completed")
            
        except Exception as e:
            print(f"[WARN] Model warm-up failed: {e}")

    def _safe_crop(self, image: np.ndarray, 
                   x1: int, y1: int, x2: int, y2: int, 
                   pad: int = 10) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        """Safe cropping utility"""
        h, w = image.shape[:2]
        x1_c = max(0, x1 - pad)
        y1_c = max(0, y1 - pad)
        x2_c = min(w, x2 + pad)
        y2_c = min(h, y2 + pad)
        
        if x2_c <= x1_c or y2_c <= y1_c:
            return image[0:0, 0:0], (x1, y1, x2, y2)
        
        return image[y1_c:y2_c, x1_c:x2_c], (x1, y1, x2, y2)

    def load_face_database(self):
        """Load FAISS database và labels từ training"""
        try:
            print(f"[INFO] Loading face database...")

            # Resolve paths using importlib.resources
            faiss_path = pkg_resources.files("lib.model") / self._faiss_name
            labels_path = pkg_resources.files("lib.model") / self._labels_name
            
            print(f"  - FAISS index: {faiss_path}")
            print(f"  - Labels: {labels_path}")
            
            # --- Load FAISS index ---
            if not faiss_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
            self.faiss_index = faiss.read_index(str(faiss_path))

            # Load labels.pkl
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels file not found: {labels_path}")
            
            with open(labels_path, "rb") as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"Labels file {labels_path} invalid format")

                self.labels = data.get("labels")
                self.use_pca = data.get("use_pca", False)
                self.pca_model = data.get("pca_model", None)
                self.index_type = data.get("index_type", "flat")
                self.ivf_nlist = data.get("ivf_nlist", 100)
                self.class_map = data.get("class_map", {})

            if self.labels is None:
                raise ValueError(f"Labels missing in {labels_path}")
            
            self.names = list(self.class_map.values()) if self.class_map else \
                [f"person_{i}" for i in range(len(self.labels))]
            self.database_loaded = True
            
            # Auto-detect similarity metric
            if isinstance(self.faiss_index, faiss.IndexFlatIP):
                self.similarity_metric = "cosine"
                self.sim_threshold = 0.6
                print("[INFO] Using cosine similarity")
            elif isinstance(self.faiss_index, faiss.IndexFlatL2):
                self.similarity_metric = "l2"
                self.sim_threshold = 0.8
                print("[INFO] Using L2 distance")
            else:
                self.similarity_metric = "cosine"
                self.sim_threshold = 0.6
                print("[WARN] Unknown FAISS index type, defaulting to cosine")
            
            print(f"[SUCCESS] Face database loaded:")
            print(f"  - Index type: {type(self.faiss_index).__name__}")
            print(f"  - Similarity metric: {self.similarity_metric}")
            print(f"  - Threshold: {self.sim_threshold}")
            print(f"  - Index size: {self.faiss_index.ntotal} embeddings")
            print(f"  - Labels: {len(self.labels)} entries")
            print(f"  - People: {len(self.names)} unique persons")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load face models: {e}")
            return False

    def detect(self, image: np.ndarray) -> List[FaceResult]:
        """Main detection method"""
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Detect faces using MTCNN
            boxes, probs = self.mtcnn.detect(pil_image)
            
            if boxes is None:
                return []
            
            # Extract aligned faces
            faces = self.mtcnn(pil_image)
            if faces is None:
                return []
            
            # Process faces
            candidates: List[Tuple[float, FaceResult]] = []
            
            for box, prob, face_tensor in zip(boxes, probs, faces):
                if face_tensor is None:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                
                result = self._process_single_face(box, prob, face_tensor)
                if result:
                    candidates.append((result.confidence, result))
            
            if not candidates:
                return []
            
            # Update performance stats
            detection_time = time.time() - start_time
            all_results = [result for _, result in candidates]
            self._update_stats(all_results, detection_time)
            
            # Return best result
            best_conf, best_result = max(candidates, key=lambda t: t[0])
            return [best_result]
            
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return []

    def _process_single_face(self, box: np.ndarray, prob: float, face_tensor: torch.Tensor) -> Optional[FaceResult]:
        """Process single detected face"""
        try:
            x1, y1, x2, y2 = map(int, box)
            detection_confidence = float(prob)
            
            # Calculate embedding
            with torch.no_grad():
                face_tensor = face_tensor.unsqueeze(0)
                embedding = self.resnet(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)

            # Apply PCA nếu có
            if getattr(self, "use_pca", False) and getattr(self, "pca_model", None) is not None:
                embedding = self.pca_model.transform([embedding])[0]
                embedding = embedding / np.linalg.norm(embedding)
            
            # Face recognition
            person_name = "Unknown"
            recognition_confidence = 0.0
            
            if self.database_loaded:
                person_name, recognition_confidence = self._recognize_face(embedding)
            
            # Calculate final confidence
            final_confidence = recognition_confidence if person_name != "Unknown" else detection_confidence

            # Create result
            return FaceResult(
                detection_type='face',
                bbox=[x1, y1, x2, y2],
                confidence=final_confidence,
                text=person_name,
                detection_confidence=detection_confidence,
                recognition_confidence=recognition_confidence,
                person_name=person_name
            )

        except Exception as e:
            print(f"[ERROR] Face processing failed: {e}")
            return None

    def _recognize_face(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Face recognition using FAISS database"""
        if not self.database_loaded or self.faiss_index is None:
            return "Unknown", 0.0
        
        try:
            query_embedding = embedding.astype('float32').reshape(1, -1)
            
            # Only normalize if using cosine similarity
            if self.similarity_metric == "cosine":
                faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(query_embedding, k=1)
            
            distance = float(distances[0][0])
            index = int(indices[0][0])
            
            # Convert distance to confidence based on metric
            if self.similarity_metric == "cosine":
                confidence = distance
                threshold_check = distance >= self.sim_threshold
            else:
                confidence = max(0.0, 1.0 - (distance / 2.0))
                threshold_check = distance <= self.sim_threshold
            
            # Apply threshold
            if threshold_check and 0 <= index < len(self.labels):
                person_label = self.labels[index]
                
                if self.class_map and person_label in self.class_map:
                    person_name = self.class_map[person_label]
                else:
                    person_name = person_label
                return person_name, confidence
            else:
                return "Unknown", confidence
                
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")
            return "Unknown", 0.0

    def _update_stats(self, results: List[FaceResult], detection_time: float):
        """Update performance statistics"""
        self.stats['total_detections'] += len(results)
        
        for result in results:
            if result.person_name and result.person_name != "Unknown":
                self.stats['known_faces'] += 1
            else:
                self.stats['unknown_faces'] += 1
        
        total_detections = self.stats['total_detections']
        if total_detections > 0:
            total_time = self.stats['avg_detection_time'] * (total_detections - len(results))
            self.stats['avg_detection_time'] = (total_time + detection_time) / total_detections

    def get_database_info(self) -> dict:
        """Get loaded database information"""
        if not self.database_loaded:
            return {"status": "No database loaded"}
        
        return {
            "status": "Database loaded",
            "faiss_total": self.faiss_index.ntotal if self.faiss_index else 0,
            "labels_count": len(self.labels) if self.labels is not None else 0,
            "people_count": len(self.names),
            "people_names": self.names,
            "similarity_metric": self.similarity_metric,
            "sim_threshold": self.sim_threshold,
            "model_name": self._model_name
        }

    def get_performance_stats(self) -> dict:
        """Get detection performance statistics"""
        total = self.stats['total_detections']
        return {
            **self.stats,
            "database_loaded": self.database_loaded,
            "recognition_rate": (self.stats['known_faces'] / total * 100) if total > 0 else 0.0,
            "model_backend": "FaceNet-PyTorch",
            "similarity_metric": self.similarity_metric
        }

    def set_sim_threshold(self, threshold: float):
        """Update similarity threshold"""
        if self.similarity_metric == "cosine" and 0.0 <= threshold <= 1.0:
            self.sim_threshold = threshold
            print(f"[INFO] Cosine similarity threshold updated to: {threshold}")
        elif self.similarity_metric == "l2" and 0.0 <= threshold <= 2.0:
            self.sim_threshold = threshold
            print(f"[INFO] L2 distance threshold updated to: {threshold}")
        else:
            print(f"[ERROR] Invalid threshold for {self.similarity_metric} metric")

    def extract_embedding_only(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding từ single face image"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            face = self.mtcnn(pil_image)
            if face is None:
                return None
            
            with torch.no_grad():
                embedding = self.resnet(face.unsqueeze(0))
                embedding = embedding.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"[ERROR] Embedding extraction failed: {e}")
            return None
