"""
Script 3: Real-Time Face Recognition Attendance System.
Run in VS Code or Terminal:
    python scripts/3_attendance_system.py
"""

import sys
import time
from pathlib import Path
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.camera import CameraStream
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.attendance_manager import AttendanceManager
from core.config import DEFAULT_CAMERA_INDEX, DEFAULT_IP_CAM_URL


def main():
    print("=" * 60)
    print("      SMART ATTENDANCE SYSTEM - LIVE CAMERA TRACKER")
    print("=" * 60)

    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    attendance_mgr = AttendanceManager()

    if not recognizer.is_loaded:
        print("\n[WARNING] Recognition model not found or not trained!")
        print("Please run 'python scripts/2_train_model.py' first to train the model.")
        return

    registered_users = recognizer.get_registered_persons()
    print(f"\n[INFO] Loaded model for {len(registered_users)} person(s): {', '.join(registered_users)}")

    print("\nSelect Camera Source:")
    print("  [1] Local USB Webcam (Default)")
    print(f"  [2] IP Camera / Phone Camera (e.g. {DEFAULT_IP_CAM_URL})")
    cam_choice = input("Enter choice (1 or 2) [default: 1]: ").strip()

    if cam_choice == "2":
        ip_url = input(f"Enter IP Webcam URL [default: {DEFAULT_IP_CAM_URL}]: ").strip()
        camera_source = ip_url if ip_url else DEFAULT_IP_CAM_URL
    else:
        cam_idx = input(f"Enter webcam index [default: {DEFAULT_CAMERA_INDEX}]: ").strip()
        camera_source = int(cam_idx) if cam_idx.isdigit() else DEFAULT_CAMERA_INDEX

    print(f"\n[INFO] Connecting to camera: {camera_source}...")
    camera = CameraStream(source=camera_source)

    if not camera.is_opened():
        print(f"[ERROR] Could not connect to camera source: {camera_source}")
        return

    window_name = "Smart Attendance System (Press 'q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    recent_alert = ""
    alert_expiry = 0

    print("[INFO] Camera started. Look at the camera to log attendance.")
    print("[INFO] Press 'q' or 'ESC' to stop and save logs.\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            display_frame = frame.copy()
            faces = detector.detect_faces(frame)

            for bbox in faces:
                x, y, w, h = bbox
                face_crop = detector.extract_face(frame, bbox)

                if face_crop is not None:
                    name, conf, is_known = recognizer.predict(face_crop)

                    if is_known:
                        box_color = (0, 200, 0)  # Green for recognized
                        # Log or update attendance
                        log_res = attendance_mgr.mark_attendance(name)
                        if log_res["status"] in ("new", "updated"):
                            recent_alert = log_res["message"]
                            alert_expiry = time.time() + 4.0  # Show for 4 seconds
                            print(f"[LOG] {recent_alert}")
                    else:
                        box_color = (0, 0, 255)  # Red for unknown

                    detector.draw_styled_box(
                        display_frame,
                        bbox,
                        label=name,
                        color=box_color,
                        confidence=conf
                    )

            # Draw HUD Header
            today_summary = attendance_mgr.get_today_summary()
            hud_text = f"Date: {today_summary['date']} | Total Present Today: {today_summary['total_present']}"
            cv2.rectangle(display_frame, (10, 10), (display_frame.shape[1] - 10, 45), (30, 30, 30), cv2.FILLED)
            cv2.putText(
                display_frame,
                hud_text,
                (20, 34),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Draw Check-in Alert Banner if active
            if time.time() < alert_expiry and recent_alert:
                cv2.rectangle(
                    display_frame,
                    (10, display_frame.shape[0] - 50),
                    (display_frame.shape[1] - 10, display_frame.shape[0] - 10),
                    (0, 150, 0),
                    cv2.FILLED
                )
                cv2.putText(
                    display_frame,
                    recent_alert,
                    (25, display_frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Camera stopped.")
        print(f"[INFO] Attendance saved to: {attendance_mgr.csv_path}")


if __name__ == "__main__":
    main()
