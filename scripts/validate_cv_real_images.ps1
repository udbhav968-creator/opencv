# ==============================================================================
# OPENCV DRS ENGINE - DEEP REAL-IMAGE VALIDATION BENCHMARK (PowerShell Script)
# Validates trained deep vision models against real COCO / Roboflow test images
# ==============================================================================

param (
    [string]$Dataset = "coco_cricket_vision",
    [string]$CheckpointDir = "models/checkpoints"
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " 🚀 OPENCV DEEP REAL-IMAGE VALIDATION BENCHMARK" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Environment Verification
Write-Host "`n[1/3] Verifying Deep Learning Validation Dependencies..." -ForegroundColor Yellow
python -m pip install --quiet torch torchvision opencv-python ultralytics

# 2. Run Real-Image Benchmark Validation
Write-Host "`n[2/3] Evaluating Models against 5,000 Real Test Images ($Dataset)..." -ForegroundColor Yellow
python scripts/validate_cv_real_images.py --dataset $Dataset --checkpoint_dir $CheckpointDir

# 3. Display Validation Report Summary
Write-Host "`n[3/3] Inspecting Generated Validation Report..." -ForegroundColor Yellow
if (Test-Path "$CheckpointDir/cv_validation_report.json") {
    Write-Host "`n✅ SUCCESS: Validation Report generated at '$CheckpointDir/cv_validation_report.json'!" -ForegroundColor Green
    Get-Content "$CheckpointDir/cv_validation_report.json"
} else {
    Write-Host "`n❌ ERROR: Validation Report not found!" -ForegroundColor Red
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " 🎉 REAL-IMAGE DEEP VALIDATION COMPLETE!" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan
