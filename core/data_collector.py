"""
Data Collector module: Captures face crops from camera and saves them to 'proje/'.
"""

import time
from pathlib import Path
import cv2
from core.camera import CameraStream
from core.config import PROJE_DIR, FACE_IMG_SIZE
from core.detector import FaceDetector


class DataCollector:
    def __init__(self, output_dir=PROJE_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector = FaceDetector()

    def collect_from_stream(self, name, num_samples=100, source=0, on_progress=None):
        """
        Interactive camera capture loop.
        Saves num_samples face crops for 'name'.
        """
        cleaned_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip()
        if not cleaned_name:
            return {"success": False, "message": "Invalid name provided."}

        camera = CameraStream(source=source)
        if not camera.is_opened():
            return {
                "success": False,
                "message": f"Could not connect to camera source: {source}"
            }

        collected_count = 0
        window_name = f"Capturing Samples for: {cleaned_name} (Press 'q' to stop)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while collected_count < num_samples:
                success, frame = camera.read()
                if not success or frame is None:
                    time.sleep(0.05)
                    continue

                display_frame = frame.copy()
                faces = self.detector.detect_faces(frame)

                if len(faces) > 0:
                    # Select largest face in frame
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face

                    face_crop = self.detector.extract_face(frame, largest_face, target_size=FACE_IMG_SIZE)

                    if face_crop is not None:
                        file_path = self.output_dir / f"{cleaned_name}_{collected_count}.jpg"
                        cv2.imwrite(str(file_path), face_crop)
                        collected_count += 1

                        if on_progress:
                            on_progress(collected_count, num_samples)

                        # Highlight detected face box in green
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Show progress counter on video
                status_text = f"Captured: {collected_count}/{num_samples} ({collected_count/num_samples*100:.0f}%)"
                cv2.putText(
                    display_frame,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    display_frame,
                    "Keep looking at camera. Press 'q' to abort.",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()

        return {
            "success": collected_count >= min(10, num_samples),
            "samples_collected": collected_count,
            "target_samples": num_samples,
            "name": cleaned_name,
            "message": f"Successfully captured {collected_count}/{num_samples} samples for {cleaned_name}."
        }
