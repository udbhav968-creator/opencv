@echo off
echo =======================================================
echo          ICC HIGH-LEVEL PRO TRAINING PIPELINE
echo =======================================================

echo [1/3] Training YOLOv8 Medium (High Accuracy)...
python %~dp0train_yolo.py --model yolov8m.pt --epochs 150 --imgsz 1280 --batch 16 --device 0

echo [2/3] Training YOLOv5 Medium...
python %~dp0train_yolo.py --model yolov5m.pt --epochs 150 --imgsz 1280 --batch 16 --device 0

echo [3/3] Training EfficientDet D1...
python %~dp0train_yolo.py --model efficientdet_d1 --epochs 150 --imgsz 1280 --batch 8 --device 0

