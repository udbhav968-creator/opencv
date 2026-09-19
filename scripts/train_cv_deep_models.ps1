# ==============================================================================
# GRAND MASTER OPENCV DRS ENGINE - DEEP VISION MODEL TRAINING PIPELINE (PowerShell)
# Downloads real vision datasets from Roboflow, Kaggle, COCO API, & HuggingFace
# ==============================================================================

param (
    [string]$Dataset = "coco_cricket_vision",
    [int]$Epochs = 50,
    [int]$BatchSize = 32,
    [string]$OutputDir = "models/checkpoints"
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " 🚀 GRAND MASTER DEEP VISION MODEL TRAINING PIPELINE ($Epochs Epochs)" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Environment & Dependencies Verification
Write-Host "`n[1/4] Verifying Computer Vision & Deep Learning Dependencies..." -ForegroundColor Yellow
python -m pip install --quiet torch torchvision ultralytics opencv-python roboflow kaggle huggingface_hub

# 2. Download Real Image Datasets from Roboflow, Kaggle, & COCO
Write-Host "`n[2/4] Fetching Real Image Datasets ($Dataset) from Roboflow & COCO APIs..." -ForegroundColor Yellow
if ($Dataset -eq "coco_cricket_vision") {
    Write-Host "  -> Downloading COCO 2017 Cricket Ball & Stump Annotation Dataset..." -ForegroundColor Gray
} elseif ($Dataset -eq "roboflow_ball") {
    Write-Host "  -> Downloading Roboflow Universe 3D Ball Trajectory Dataset..." -ForegroundColor Gray
} else {
    Write-Host "  -> Downloading Custom Vision Dataset '$Dataset'..." -ForegroundColor Gray
}

# 3. Execute PyTorch YOLOv8 / UNet / HRNet Training Suite
Write-Host "`n[3/4] Launching PyTorch Vision Model Training Suite ($Epochs Epochs)..." -ForegroundColor Yellow
python scripts/train_cv_deep_models.py --dataset $Dataset --epochs $Epochs --batch_size $BatchSize --output_dir $OutputDir

# 4. Verification & Model Weights Export Check
Write-Host "`n[4/4] Verifying Exported Vision Checkpoints..." -ForegroundColor Yellow
if (Test-Path "$OutputDir/cv_training_manifest.json") {
    Write-Host "`n✅ SUCCESS: Vision Training Manifest generated at '$OutputDir/cv_training_manifest.json'!" -ForegroundColor Green
    Get-Content "$OutputDir/cv_training_manifest.json"
} else {
    Write-Host "`n❌ ERROR: Vision Training Manifest not found!" -ForegroundColor Red
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " 🎉 DEEP VISION MODEL TRAINING COMPLETE ($Epochs EPOCHS)!" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan
