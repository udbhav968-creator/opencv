# dataset_harvester.py
"""
Dataset Harvester for ICC Cricket Ball Detection

Downloads public cricket-ball datasets from Kaggle, optionally clones GitHub repositories containing images
and converts them to the YOLO format used by the training pipeline.

Usage:
    python dataset_harvester.py [--dry-run] [--kaggle <dataset_name>] [--github <search_query>]
"""

import os
import argparse
import subprocess
import sys
import shutil
import json
from pathlib import Path

# Project directories (relative to this file)
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL = DATASET_DIR / "labels" / "val"

def ensure_dirs():
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)

def download_kaggle(dataset_name: str, dest: Path) -> bool:
    """Download a Kaggle dataset using the Kaggle CLI.
    Returns True on success, False otherwise.
    """
    print(f"[Harvester] Attempting Kaggle download: {dataset_name}")
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "--unzip", "-p", str(dest)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[Harvester] Kaggle CLI error: {res.stderr.strip()}")
            return False
        print("[Harvester] Kaggle dataset downloaded successfully.")
        return True
    except FileNotFoundError:
        print("[Harvester] Kaggle CLI not installed.")
        return False
    except Exception as e:
        print(f"[Harvester] Unexpected error: {e}")
        return False

def copy_images_and_labels(src_root: Path, split: str = "train"):
    """Copy image files and matching YOLO .txt label files from src_root into
    the project's dataset structure.
    """
    img_dest = IMAGES_TRAIN if split == "train" else IMAGES_VAL
    lbl_dest = LABELS_TRAIN if split == "train" else LABELS_VAL
    for root, _, files in os.walk(src_root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = Path(root) / f
                stem = img_path.stem
                label_path = Path(root) / f"{stem}.txt"
                if label_path.is_file():
                    shutil.copy2(img_path, img_dest / img_path.name)
                    shutil.copy2(label_path, lbl_dest / label_path.name)

def harvest_kaggle(dataset_name: str, dry_run: bool = False):
    temp_dir = BASE_DIR / "_kaggle_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    if not download_kaggle(dataset_name, temp_dir):
        print("[Harvester] Kaggle download failed – skipping.")
        return
    if dry_run:
        print("[Harvester] Dry‑run: would copy images/labels from Kaggle dataset.")
        return
    # Simple 90/10 split for train/val
    all_items = list(temp_dir.rglob('*.jpg')) + list(temp_dir.rglob('*.png'))
    total = len(all_items)
    train_cut = int(total * 0.9)
    for i, img_path in enumerate(all_items):
        split = "train" if i < train_cut else "val"
        copy_images_and_labels(img_path.parent, split)
    print(f"[Harvester] Copied {total} images from Kaggle dataset.")

def harvest_github(search_query: str, max_repos: int = 5, dry_run: bool = False):
    """Search GitHub for repositories containing cricket ball datasets and copy any
    YOLO‑style image/label pairs they may contain.
    """
    import requests
    url = f"https://api.github.com/search/repositories?q={search_query}+in:name,description&per_page={max_repos}"
    print(f"[Harvester] Querying GitHub: {url}")
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[Harvester] GitHub API error: {resp.status_code}")
        return
    data = resp.json()
    for item in data.get('items', []):
        clone_url = item['clone_url']
        repo_name = item['full_name'].replace('/', '-')
        repo_dir = BASE_DIR / f"_github_tmp_{repo_name}"
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        if dry_run:
            print(f"[Harvester] Dry‑run: would clone {clone_url}")
            continue
        print(f"[Harvester] Cloning {clone_url} ...")
        subprocess.run(["git", "clone", "--depth", "1", clone_url, str(repo_dir)], check=False)
        copy_images_and_labels(repo_dir, "train")
    print(f"[Harvester] Processed {max_repos} GitHub repositories.")

def harvest_all_sources(dry_run: bool = False):
    """
    Harvests datasets from ALL authentic sources:
      - Kaggle 1: udbhav968/cricket-ball-detection
      - Kaggle 2: coco-class32/sports-ball
      - Kaggle 3: praveengovi/cricket-ball-tracking-dataset
      - Kaggle 4: cric-ai/icc-cricket-ball-and-pitch
      - GitHub Search API: Top 10 public repositories matching 'cricket ball dataset'
    """
    kaggle_datasets = [
        "udbhav968/cricket-ball-detection",
        "coco-class32/sports-ball",
        "praveengovi/cricket-ball-tracking-dataset",
        "cric-ai/icc-cricket-ball-and-pitch"
    ]

    print("[Harvester] Starting Multi-Source Harvester (Kaggle + GitHub + Open APIs)...")
    for ds in kaggle_datasets:
        harvest_kaggle(ds, dry_run=dry_run)

    harvest_github("cricket ball dataset", max_repos=10, dry_run=dry_run)
    print("[Harvester] Multi-Source Dataset Harvesting Complete!")


def main():
    parser = argparse.ArgumentParser(description="ICC Universal Multi-Source Dataset Harvester")
    parser.add_argument('--dry-run', action='store_true', help='Only simulate actions')
    parser.add_argument('--full', action='store_true', default=True, help='Run full harvest (default)')
    args = parser.parse_args()

    ensure_dirs()
    harvest_all_sources(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
