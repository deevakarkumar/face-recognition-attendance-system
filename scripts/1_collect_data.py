"""
Script 1: Register a new user and capture face samples.
Run in VS Code or Terminal:
    python scripts/1_collect_data.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_collector import DataCollector
from core.trainer import ModelTrainer
from core.config import DEFAULT_CAMERA_INDEX, DEFAULT_IP_CAM_URL


def main():
    print("=" * 60)
    print("      SMART ATTENDANCE SYSTEM - USER REGISTRATION")
    print("=" * 60)

    name = input("\nEnter the person's name (e.g. Alice, John): ").strip()
    if not name:
        print("[Error] Name cannot be empty.")
        return

    samples_input = input("Number of face samples to capture [default: 100]: ").strip()
    num_samples = int(samples_input) if samples_input.isdigit() else 100

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

    print("\n[INFO] Starting camera. Look at the camera and rotate head slightly.")
    print("[INFO] Press 'q' on the camera window to abort at any time.\n")

    collector = DataCollector()
    result = collector.collect_from_stream(name=name, num_samples=num_samples, source=camera_source)

    if result["success"]:
        print(f"\n[SUCCESS] {result['message']}")
        retrain = input("\nDo you want to retrain the recognition model now? (Y/n): ").strip().lower()
        if retrain != "n":
            print("[INFO] Retraining model...")
            train_res = ModelTrainer().train_lbph()
            print(f"[SUCCESS] {train_res['message']}")
    else:
        print(f"\n[WARNING] {result['message']}")


if __name__ == "__main__":
    main()
