import cv2
import numpy as np
from typing import List
from .base_preprocessor import BasePreprocessor

class FacePreprocessor(BasePreprocessor):
    """Preprocessor nâng cao cho khuôn mặt.
       Sinh nhiều biến thể ảnh để tăng hiệu quả detect + recognition (MTCNN + FaceNet).
    """

    def __init__(self, align: bool = False):
        super().__init__('face')
        self.min_size = 160
        self.align = align

    def face_preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        processed_variants = []
        try:
            # 1. Chuẩn hóa ảnh input
            rgb = self._validate_image(image, force_gray=False)  # giữ RGB

            h, w = rgb.shape[:2]
            if h < self.min_size or w < self.min_size:
                rgb = cv2.resize(rgb, (self.min_size, self.min_size), interpolation=cv2.INTER_CUBIC)

            processed_variants.append(rgb)

            # 2. CLAHE (cân bằng sáng cục bộ)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            clahe_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            processed_variants.append(clahe_rgb)

            # 3. Unsharp Mask (tăng chi tiết nếu ảnh mờ)
            gaussian = cv2.GaussianBlur(clahe_rgb, (3,3), 0)
            unsharp = cv2.addWeighted(clahe_rgb, 1.5, gaussian, -0.5, 0)
            processed_variants.append(np.clip(unsharp, 0, 255).astype(np.uint8))

            # 4. Gamma correction (sáng và tối)
            for gamma in [0.7, 1.2]:
                gamma_corrected = np.array(
                    255 * ((clahe_rgb / 255.0) ** gamma),
                    dtype='uint8'
                )
                processed_variants.append(gamma_corrected)

            # 5. Optional: Face alignment
            if self.align:
                # Ở đây bạn có thể implement align bằng landmarks (mắt, mũi, miệng)
                aligned = self._dummy_align(rgb)
                processed_variants.append(aligned)

            return processed_variants

        except Exception as e:
            print(f"[ERROR] Face preprocessing failed: {e}")
            return [rgb] if 'rgb' in locals() else []

    def _dummy_align(self, img: np.ndarray) -> np.ndarray:
        """Placeholder align function (xoay khuôn mặt theo mắt).
           Cần landmarks để làm align chính xác, tạm thời trả ảnh gốc.
        """
        return img

    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        return self.face_preprocess(image)