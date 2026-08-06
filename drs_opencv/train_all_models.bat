@echo off
echo =======================================================
echo          ICC ULTRA DEEP LEARNING PRO TRAINING PIPELINE (500 EPOCHS)
echo =======================================================

echo [1/3] Training YOLOv8 Extra Large (500 Epochs, 1280px)...
python %~dp0train_yolo.py --model yolov8x.pt --epochs 500 --imgsz 1280 --batch 16 --device 0

echo [2/3] Training YOLOv5 Extra Large (500 Epochs, 1280px)...
python %~dp0train_yolo.py --model yolov5x.pt --epochs 500 --imgsz 1280 --batch 16 --device 0

echo [3/3] Training EfficientDet D3 Deep (500 Epochs, 1280px)...
python %~dp0train_yolo.py --model efficientdet_d3 --epochs 500 --imgsz 1280 --batch 8 --device 0
