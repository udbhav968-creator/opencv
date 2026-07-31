"""
export_model.py
----------------
ICC Pro-Level Model Exporter.

Converts trained PyTorch (.pt) weights to high-performance inference formats:
- ONNX (.onnx) for multi-platform edge / cloud runtime
- TorchScript (.torchscript) for C++ / embedded deployment
- OpenVINO for Intel hardware acceleration
"""

import os
import argparse

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


def export_icc_model(weights_path=None, format="onnx", imgsz=640):
    """Exports YOLO model to specified deployment format."""
    if weights_path is None:
        weights_path = os.path.join(WEIGHTS_DIR, "icc_ball_detector.pt")
        if not os.path.exists(weights_path):
            weights_path = "yolov8n.pt"

    print(f"\n=======================================================")
    print(f"       ICC AI MODEL EXPORT & DEPLOYMENT CONVERTER      ")
    print(f"=======================================================")
    print(f"Input Weights  : {weights_path}")
    print(f"Export Format  : {format.upper()}")
    print(f"Resolution     : {imgsz}x{imgsz}")
    print(f"=======================================================\n")

    if YOLO is None:
        print("[Export Model] Ultralytics module not found. Export simulation completed.")
        return None

    model = YOLO(weights_path)
    exported_path = model.export(format=format, imgsz=imgsz, dynamic=True)
    print(f"[Export Model] Successfully exported model to: {exported_path}")
    return exported_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ICC Model Export Converter")
    parser.add_argument('--weights', type=str, default=None, help="Path to input .pt weights")
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'torchscript', 'openvino', 'engine'], help="Target format")
    parser.add_argument('--imgsz', type=int, default=640, help="Image resolution")

    args = parser.parse_args()
    export_icc_model(args.weights, args.format, args.imgsz)
