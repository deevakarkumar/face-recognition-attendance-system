"""
Script 2: Train the Face Recognition Model from all photos in 'proje/'.
Run in VS Code or Terminal:
    python scripts/2_train_model.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.trainer import ModelTrainer


def main():
    print("=" * 60)
    print("      SMART ATTENDANCE SYSTEM - MODEL TRAINING")
    print("=" * 60)

    trainer = ModelTrainer()
    print("\n[INFO] Scanning dataset directory for face images...")
    faces, ids, id_to_name = trainer.prepare_dataset()

    if not faces:
        print("[ERROR] No face images found in the 'proje/' directory!")
        print("Please run 'python scripts/1_collect_data.py' first to register users.")
        return

    print(f"[INFO] Found {len(faces)} total face images across {len(id_to_name)} registered person(s):")
    for pid, pname in id_to_name.items():
        count = sum(1 for i in ids if i == pid)
        print(f"   - {pname}: {count} samples")

    print("\n[INFO] Training LBPH Face Recognition Model...")
    result = trainer.train_lbph()

    if result["success"]:
        print(f"\n[SUCCESS] {result['message']}")
        print(f"Model saved to: data/models/lbph_model.xml")
        print(f"Labels saved to: data/models/labels.json")
    else:
        print(f"\n[ERROR] Training failed: {result['message']}")

    # If 2 or more classes exist, offer CNN training
    if len(id_to_name) >= 2:
        train_cnn_opt = input("\nWould you also like to train the Keras CNN model? (y/N): ").strip().lower()
        if train_cnn_opt == "y":
            print("[INFO] Training Keras CNN model (this may take a moment)...")
            cnn_res = trainer.train_cnn()
            print(f"[RESULT] {cnn_res['message']}")


if __name__ == "__main__":
    main()
