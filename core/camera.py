"""
Camera stream handler supporting both local USB webcams and IP cameras (HTTP / RTSP).
"""

import time
import urllib.request
import cv2
import numpy as np


class CameraStream:
    def __init__(self, source=0):
        """
        source: can be an integer (0, 1 for local webcam)
                or a string URL (e.g. 'http://10.0.1.4:8080/shot.jpg' or rtsp link).
        """
        self.source = source
        self.is_ip_url = isinstance(source, str) and (
            source.startswith("http://") or source.startswith("https://")
        )
        self.cap = None

        if not self.is_ip_url:
            # Parse integer if passed as string "0"
            if isinstance(source, str) and source.isdigit():
                self.source = int(source)
            self.cap = cv2.VideoCapture(self.source)

    def is_opened(self):
        if self.is_ip_url:
            try:
                req = urllib.request.urlopen(self.source, timeout=3)
                return req.getcode() == 200
            except Exception:
                return False
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """
        Returns (success: bool, frame: np.ndarray)
        """
        if self.is_ip_url:
            try:
                response = urllib.request.urlopen(self.source, timeout=3)
                img_array = np.array(bytearray(response.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    return True, frame
                return False, None
            except Exception:
                return False, None
        else:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
