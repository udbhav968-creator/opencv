"""
train_yolo.py
-------------
ICC Pro-Level Custom YOLO Training Pipeline for Cricket Ball & Keypoint Detection.

Trains Ultralytics YOLOv8/YOLOv11 model on custom or Kaggle dataset,
logs training performance metrics, and exports weights for deployment.
"""

import os
import sys
import argparse
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("[Train YOLO] Ultralytics module not found. Install via: pip install ultralytics")
    YOLO = None

from dataset_manager import generate_synthetic_icc_dataset, YAML_PATH


WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def train_icc_yolo(
    base_model="yolov8n.pt",
    epochs=10,
    batch_size=8,
    imgsz=640,
    lr0=0.01,
    device="cpu",
    export_onnx=True,
    data_yaml=YAML_PATH
):
    """
    Executes training loop using Ultralytics YOLO framework.
    """
    if YOLO is None:
        print("[Train YOLO] Cannot run training without Ultralytics installed.")
        return None

    if not os.path.exists(data_yaml):
        print("[Train YOLO] Dataset YAML not found. Generating synthetic dataset...")
        data_yaml = generate_synthetic_icc_dataset(num_train=50, num_val=10)

    print(f"\n=======================================================")
    print(f"       ICC PRO-LEVEL AI MODEL TRAINING PIPELINE        ")
    print(f"=======================================================")
    print(f"Base Model : {base_model}")
    print(f"Dataset    : {data_yaml}")
    print(f"Epochs     : {epochs}")
    print(f"Batch Size : {batch_size}")
    print(f"Image Size : {imgsz}")
    print(f"Learning Rate: {lr0}")
    print(f"=======================================================\n")

    start_time = time.time()

    # Initialize model
    if "efficientdet" in base_model.lower():
        print("[Train YOLO] EfficientDet selected. Using torchvision fallback / mock training.")
        time.sleep(2) # Mock training time
        best_pt_path = os.path.join(WEIGHTS_DIR, "efficientdet_mock.pt")
        with open(best_pt_path, "w") as f:
            f.write("mock_weights")
        return best_pt_path
    else:
        model = YOLO(base_model)

        # Train model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            lr0=lr0,
            device=device,
            project=WEIGHTS_DIR,
            name="icc_run",
            exist_ok=True,
            verbose=True
        )

    elapsed = time.time() - start_time
    print(f"\n[Train YOLO] Training completed in {elapsed:.2f} seconds!")

    # Save best PyTorch weights
    best_pt_path = os.path.join(WEIGHTS_DIR, "icc_ball_detector.pt")
    model.save(best_pt_path)
    print(f"[Train YOLO] Saved trained PyTorch weights to: {best_pt_path}")

    # Export ONNX model if requested
    if export_onnx:
        print("[Train YOLO] Exporting model to high-speed ONNX format...")
        try:
            onnx_file = model.export(format="onnx", imgsz=imgsz, dynamic=True)
            target_onnx = os.path.join(WEIGHTS_DIR, "icc_ball_detector.onnx")
            if os.path.exists(onnx_file) and onnx_file != target_onnx:
                import shutil
                shutil.copy(onnx_file, target_onnx)
            print(f"[Train YOLO] Exported ONNX model to: {target_onnx}")
        except Exception as e:
            print(f"[Train YOLO] ONNX export note: {e}")

    return best_pt_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ICC Pro YOLO Training Script")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help="Base weights (e.g. yolov8n.pt, yolov5s.pt, efficientdet_d0)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=8, help="Batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Image resolution")
    parser.add_argument('--lr0', type=float, default=0.01, help="Initial learning rate")
    parser.add_argument('--device', type=str, default='cpu', help="Compute device (cpu, cuda, 0, etc.)")
    parser.add_argument('--no-onnx', action='store_true', help="Disable ONNX export")

    args = parser.parse_args()

    train_icc_yolo(
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        device=args.device,
        export_onnx=not args.no_onnx
    )
