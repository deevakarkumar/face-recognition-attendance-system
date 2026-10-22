"""
Face detection module using Haar Cascade with preprocessing utilities.
"""

import cv2
import numpy as np
from core.config import CASCADE_PATH, FACE_IMG_SIZE


class FaceDetector:
    def __init__(self, cascade_path=None):
        path = str(cascade_path or CASCADE_PATH)
        self.classifier = cv2.CascadeClassifier(path)
        if self.classifier.empty():
            raise FileNotFoundError(f"Could not load Haar cascade classifier from: {path}")

    def detect_faces(self, frame, scale_factor=1.3, min_neighbors=5, min_size=(60, 60)):
        """
        Detect faces in a BGR or Grayscale frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        return faces

    def extract_face(self, frame, bbox, target_size=FACE_IMG_SIZE, equalize=True):
        """
        Crop face from frame, convert to grayscale, resize, and optionally apply histogram equalization.
        Returns 2D grayscale face array (target_size[0], target_size[1]).
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        # Clamp bounding box coordinates within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        if w <= 0 or h <= 0:
            return None

        face_roi = frame[y:y+h, x:x+w]

        if len(face_roi.shape) == 3:
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_roi

        face_resized = cv2.resize(face_gray, target_size, interpolation=cv2.INTER_AREA)

        if equalize:
            face_resized = cv2.equalizeHist(face_resized)

        return face_resized

    @staticmethod
    def draw_styled_box(frame, bbox, label="Unknown", color=(0, 255, 0), confidence=None):
        """
        Draw an attractive bounding box with corner highlights and a clean label header.
        """
        x, y, w, h = bbox
        line_length = int(min(w, h) * 0.2)
        thickness = 2
        corner_thickness = 4

        # Main rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        # Top-left corner
        cv2.line(frame, (x, y), (x + line_length, y), color, corner_thickness)
        cv2.line(frame, (x, y), (x, y + line_length), color, corner_thickness)
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - line_length, y), color, corner_thickness)
        cv2.line(frame, (x + w, y), (x + w, y + line_length), color, corner_thickness)
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + line_length, y + h), color, corner_thickness)
        cv2.line(frame, (x, y + h), (x, y + h - line_length), color, corner_thickness)
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - line_length, y + h), color, corner_thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - line_length), color, corner_thickness)

        # Label text
        display_text = label
        if confidence is not None:
            display_text += f" ({confidence:.0f}%)"

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)

        # Background badge for label
        badge_top = max(0, y - text_h - 10)
        badge_bottom = y
        cv2.rectangle(frame, (x, badge_top), (x + text_w + 12, badge_bottom), color, cv2.FILLED)
        cv2.putText(
            frame,
            display_text,
            (x + 6, y - 6),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )
        return frame
