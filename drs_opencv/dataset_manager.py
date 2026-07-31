"""
dataset_manager.py
------------------
ICC Pro-Level Cricket Dataset Manager & Annotator.

Features:
1. Download official cricket ball detection datasets from Kaggle / GitHub / Open Sources.
2. Automated synthetic dataset generator for local zero-dependency YOLOv8/YOLOv11 training.
3. Generates dataset.yaml, train/val image & label splits ready for Ultralytics YOLO training.
"""

import os
import sys
import argparse
import random
import cv2
import numpy as np


DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
YAML_PATH = os.path.join(DATASET_DIR, 'dataset.yaml')


def setup_dataset_structure():
    """Creates directory structure for YOLO format dataset."""
    dirs = [
        os.path.join(DATASET_DIR, 'images', 'train'),
        os.path.join(DATASET_DIR, 'images', 'val'),
        os.path.join(DATASET_DIR, 'labels', 'train'),
        os.path.join(DATASET_DIR, 'labels', 'val'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    yaml_content = f"""# ICC Official Cricket Ball Tracking Dataset Configuration
path: {os.path.abspath(DATASET_DIR).replace('\\\\', '/')}
train: images/train
val: images/val

names:
  0: cricket_ball
"""
    with open(YAML_PATH, 'w') as f:
        f.write(yaml_content)

    print(f"[Dataset Manager] Created dataset structure and configuration at: {YAML_PATH}")


def download_kaggle_dataset(dataset_name="cricket-ball-detection"):
    """Downloads dataset from Kaggle using Kaggle API CLI if installed."""
    print(f"[Dataset Manager] Attempting to download Kaggle dataset: {dataset_name}")
    try:
        import subprocess
        cmd = f"kaggle datasets download -d {dataset_name} --unzip -p {DATASET_DIR}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print("[Dataset Manager] Kaggle dataset successfully downloaded!")
            return True
        else:
            print(f"[Dataset Manager] Kaggle CLI note: {res.stderr.strip()}")
            print("[Dataset Manager] Falling back to ICC Synthetic Dataset Generator...")
            return False
    except Exception as e:
        print(f"[Dataset Manager] Kaggle API unavailable ({e}). Using synthetic generator...")
        return False


def generate_synthetic_icc_dataset(num_train=100, num_val=20):
    """
    Generates synthetic cricket pitch images with annotated YOLO bounding boxes
    for cricket ball detection.
    """
    setup_dataset_structure()
    print(f"[Dataset Manager] Generating {num_train} train and {num_val} val ICC synthetic samples...")

    splits = [('train', num_train), ('val', num_val)]

    for split_name, count in splits:
        img_dir = os.path.join(DATASET_DIR, 'images', split_name)
        lbl_dir = os.path.join(DATASET_DIR, 'labels', split_name)

        for i in range(count):
            # Create synthetic pitch background (640x640)
            h, w = 640, 640
            img = np.zeros((h, w, 3), dtype=np.uint8)

            # Draw green grass pitch with variation
            grass_color = (random.randint(20, 40), random.randint(100, 160), random.randint(20, 50))
            img[:] = grass_color

            # Draw pitch rectangle (light brownish beige)
            pitch_color = (random.randint(120, 160), random.randint(160, 200), random.randint(170, 210))
            cv2.fillPoly(img, [np.array([[200, 50], [440, 50], [550, 600], [90, 600]])], pitch_color)

            # Draw stumps at batsman end
            cv2.rectangle(img, (310, 80), (330, 120), (230, 230, 230), -1)

            # Place cricket ball at random trajectory position
            bx = random.randint(150, 490)
            by = random.randint(100, 550)
            br = random.randint(6, 16)  # radius in px

            # Ball color (red/white/pink)
            ball_type = random.choice(['red', 'white', 'pink'])
            if ball_type == 'red':
                b_color = (random.randint(15, 45), random.randint(20, 50), random.randint(180, 245))
            elif ball_type == 'white':
                b_color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
            else:
                b_color = (random.randint(140, 180), random.randint(60, 100), random.randint(220, 255))

            # Draw ball with subpixel anti-aliasing
            cv2.circle(img, (bx, by), br, b_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (bx - 2, by - 2), max(1, br // 3), (255, 255, 255), -1, lineType=cv2.LINE_AA)  # specular highlight

            # Save image
            file_stem = f"icc_sample_{split_name}_{i:04d}"
            img_path = os.path.join(img_dir, f"{file_stem}.jpg")
            cv2.imwrite(img_path, img)

            # Save YOLO label (class_id center_x center_y width height in normalized 0..1 coordinates)
            norm_cx = bx / w
            norm_cy = by / h
            norm_bw = (br * 2.4) / w
            norm_bh = (br * 2.4) / h

            lbl_path = os.path.join(lbl_dir, f"{file_stem}.txt")
            with open(lbl_path, 'w') as f:
                f.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}\n")

    print(f"[Dataset Manager] Successfully generated dataset with {num_train + num_val} annotated samples!")
    return YAML_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ICC Pro Cricket Dataset Manager")
    parser.add_argument('--dataset', type=str, default='cricket-ball-detection', help="Kaggle dataset name")
    parser.add_argument('--generate-synthetic', action='store_true', help="Generate synthetic ICC annotated dataset")
    parser.add_argument('--train-samples', type=int, default=100, help="Number of training samples")
    parser.add_argument('--val-samples', type=int, default=20, help="Number of validation samples")
    args = parser.parse_args()

    setup_dataset_structure()
    if args.generate_synthetic or not download_kaggle_dataset(args.dataset):
        generate_synthetic_icc_dataset(num_train=args.train_samples, num_val=args.val_samples)
