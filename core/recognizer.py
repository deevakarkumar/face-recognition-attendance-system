"""
Face recognition module with confidence thresholding, dynamic label lookup,
and Unknown-face rejection.
"""

import json
from pathlib import Path
import cv2
import numpy as np

from core.config import (
    LBPH_MODEL_PATH,
    LABEL_MAP_PATH,
    LBPH_CONFIDENCE_THRESHOLD,
    FACE_IMG_SIZE,
    KERAS_MODEL_PATH
)
from core.detector import FaceDetector


class FaceRecognizer:
    def __init__(self, model_path=LBPH_MODEL_PATH, label_map_path=LABEL_MAP_PATH, threshold=LBPH_CONFIDENCE_THRESHOLD):
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.threshold = threshold
        self.recognizer = None
        self.id_to_name = {}
        self.is_loaded = False

        self.load_model()

    def load_model(self):
        """Loads trained LBPH model and label mapping."""
        if not self.model_path.exists() or not self.label_map_path.exists():
            self.is_loaded = False
            return False

        try:
            # Load label mapping
            with open(self.label_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure integer keys for lookup
                self.id_to_name = {int(k): v for k, v in data.get("id_to_name", {}).items()}

            # Load LBPH model
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(str(self.model_path))
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[Warning] Error loading face recognition model: {e}")
            self.is_loaded = False
            return False

    def predict(self, face_img):
        """
        Predicts person identity from a preprocessed face crop.
        Returns:
            label (str): Recognized name or "Unknown"
            confidence (float): Confidence score between 0.0 and 100.0%
            is_known (bool): True if recognized within threshold
        """
        if not self.is_loaded or self.recognizer is None:
            return "Model Not Trained", 0.0, False

        if face_img is None or face_img.size == 0:
            return "Unknown", 0.0, False

        # Ensure correct size and grayscale
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        if face_img.shape != FACE_IMG_SIZE:
            face_img = cv2.resize(face_img, FACE_IMG_SIZE, interpolation=cv2.INTER_AREA)

        try:
            label_id, distance = self.recognizer.predict(face_img)

            # In LBPH: 0 = exact match; values above threshold are considered "Unknown"
            # Calculate human-friendly confidence percentage (0 to 100%)
            confidence_pct = max(0.0, min(100.0, 100.0 - distance))

            if distance <= self.threshold and label_id in self.id_to_name:
                name = self.id_to_name[label_id]
                return name, confidence_pct, True
            else:
                return "Unknown", confidence_pct, False
        except Exception as e:
            print(f"[Error] Prediction failed: {e}")
            return "Unknown", 0.0, False

    def get_registered_persons(self):
        """Returns list of currently registered person names."""
        return list(self.id_to_name.values())
