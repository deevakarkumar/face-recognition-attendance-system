"""
Model training module: Consolidates image dataset from 'proje/',
builds label mappings, and trains the recognition model.
"""

import json
import os
import time
from pathlib import Path
import cv2
import numpy as np

from core.config import (
    PROJE_DIR,
    DATA_DIR,
    MODELS_DIR,
    LBPH_MODEL_PATH,
    LABEL_MAP_PATH,
    FACE_IMG_SIZE,
    KERAS_MODEL_PATH
)
from core.detector import FaceDetector


class ModelTrainer:
    def __init__(self, dataset_dir=PROJE_DIR, models_dir=MODELS_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.detector = FaceDetector()

    def prepare_dataset(self, target_size=FACE_IMG_SIZE):
        """
        Loads all images from the dataset directory.
        Expected format: {Name}_{i}.jpg
        Returns:
            faces: list of 2D uint8 numpy arrays (target_size)
            ids: list of int IDs
            label_map: dict of {int_id: name_str}
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [
            f for f in self.dataset_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions and "_" in f.stem
        ]

        if not files:
            return [], [], {}

        # Discover unique person names
        unique_names = sorted(list({f.stem.rsplit("_", 1)[0] for f in files}))
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        id_to_name = {idx: name for idx, name in enumerate(unique_names)}

        faces = []
        ids = []

        for file_path in files:
            name = file_path.stem.rsplit("_", 1)[0]
            label_id = name_to_id[name]

            # Read image
            img = cv2.imread(str(file_path))
            if img is None:
                continue

            # Ensure image is grayscale and equalized
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # If already face size, just resize & equalize
            if gray.shape != target_size:
                gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

            gray = cv2.equalizeHist(gray)
            faces.append(gray)
            ids.append(label_id)

        return faces, ids, id_to_name

    def train_lbph(self):
        """
        Trains the OpenCV LBPH Face Recognizer and saves the model + label map.
        Returns a dict of training results.
        """
        start_time = time.time()
        faces, ids, id_to_name = self.prepare_dataset()

        if not faces:
            return {
                "success": False,
                "message": f"No valid training images found in: {self.dataset_dir}",
                "num_samples": 0,
                "num_persons": 0
            }

        # Initialize LBPH Recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )

        recognizer.train(faces, np.array(ids, dtype=np.int32))

        # Save model
        recognizer.save(str(LBPH_MODEL_PATH))

        # Save label map JSON (string keys for JSON compatibility)
        label_data = {
            "id_to_name": {str(k): v for k, v in id_to_name.items()},
            "name_to_id": {v: k for k, v in id_to_name.items()},
            "num_persons": len(id_to_name),
            "num_samples": len(faces),
            "persons": list(id_to_name.values()),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=2)

        elapsed = time.time() - start_time
        return {
            "success": True,
            "message": f"Model successfully trained for {len(id_to_name)} person(s) on {len(faces)} images in {elapsed:.2f}s.",
            "num_samples": len(faces),
            "num_persons": len(id_to_name),
            "persons": list(id_to_name.values()),
            "duration": round(elapsed, 2)
        }

    def train_cnn(self, epochs=10, batch_size=16):
        """
        Trains a Keras CNN model when at least 2 distinct classes exist.
        """
        faces, ids, id_to_name = self.prepare_dataset()
        num_classes = len(id_to_name)

        if num_classes < 2:
            return {
                "success": False,
                "message": (
                    f"CNN softmax classification requires at least 2 people (found {num_classes}). "
                    "Register at least 2 users or use the LBPH model."
                )
            }

        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
            from tensorflow.keras.utils import to_categorical

            X = np.array(faces, dtype=np.float32).reshape(-1, FACE_IMG_SIZE[0], FACE_IMG_SIZE[1], 1) / 255.0
            y = to_categorical(np.array(ids), num_classes=num_classes)

            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(FACE_IMG_SIZE[0], FACE_IMG_SIZE[1], 1)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(num_classes, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.15, verbose=0)
            model.save(str(KERAS_MODEL_PATH))

            return {
                "success": True,
                "message": f"CNN model trained on {len(faces)} images across {num_classes} classes."
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to train CNN model: {e}"
            }
