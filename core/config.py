"""
Configuration settings for Smart Face Recognition Attendance System.
All paths are dynamically resolved relative to the project root directory.
"""

import os
from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Directories
DATA_DIR = BASE_DIR / "data"
PROJE_DIR = BASE_DIR / "proje"
MODELS_DIR = DATA_DIR / "models"
RECORDS_DIR = BASE_DIR / "records"

# Ensure necessary directories exist
DATA_DIR.mkdir(exist_ok=True)
PROJE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RECORDS_DIR.mkdir(exist_ok=True)

# Files
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
ATTENDANCE_CSV = BASE_DIR / "attendance.csv"
ATTENDANCE_DB = DATA_DIR / "attendance.db"

# Model Artifacts
LBPH_MODEL_PATH = MODELS_DIR / "lbph_model.xml"
LABEL_MAP_PATH = MODELS_DIR / "labels.json"
KERAS_MODEL_PATH = BASE_DIR / "final_model.h5"

# Recognition Parameters
FACE_IMG_SIZE = (100, 100)
LBPH_CONFIDENCE_THRESHOLD = 75.0  # In LBPH, lower distance means higher confidence (<= 75 is recognized)
CNN_CONFIDENCE_THRESHOLD = 0.70   # For CNN softmax (>= 70% is recognized)

# Attendance System Defaults
COOLDOWN_SECONDS = 300  # 5 minutes cooldown before re-logging same person on entry
DEFAULT_CAMERA_INDEX = 0  # 0 for default USB webcam
DEFAULT_IP_CAM_URL = "http://10.0.1.4:8080/shot.jpg"
