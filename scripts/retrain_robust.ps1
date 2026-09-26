# Retrain CoffeeGuard with Enhanced Robustness
# Addresses poor generalization on bright/high-res/PNG web images
# 
# This script:
# 1. Trains with aggressive augmentation (configs/train/effnetv2_b0_robust.yaml)
# 2. Exports the model bundle
# 3. Fits OOD gates with relaxed thresholds
# 4. Evaluates robustness
#
# Usage: .\scripts\retrain_robust.ps1 [seed]

param(
    [int]$Seed = 42,
    [switch]$SkipUpload,
    [switch]$LocalOnly
)

Write-Host "=== CoffeeGuard Robust Retraining Pipeline ===" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ConfigFile = "configs/train/effnetv2_b0_robust.yaml"
$BundleName = "coffeeguard-effv2b0-v1-robust"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunName = "${Timestamp}-effnetv2_b0_robust-s${Seed}"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Config: $ConfigFile"
Write-Host "  Seed: $Seed"
Write-Host "  Expected Run: runs/$RunName"
Write-Host "  Target Bundle: artifacts/models/$BundleName"
Write-Host ""

# Step 1: Verify data is ready
Write-Host "[1/7] Verifying dataset..." -ForegroundColor Cyan
if (-not (Test-Path "data/processed")) {
    Write-Host "  Data not found. Running data preparation..." -ForegroundColor Yellow
    uv run coffeeguard data download
    uv run coffeeguard data prepare
    uv run coffeeguard data report
}
Write-Host "  ✓ Dataset ready" -ForegroundColor Green
Write-Host ""

# Step 2: Upload data to Kaggle (if training remotely)
if (-not $LocalOnly -and -not $SkipUpload) {
    Write-Host "[2/7] Uploading data to Kaggle..." -ForegroundColor Cyan
    try {
        uv run coffeeguard remote upload-data
        Write-Host "  ✓ Data uploaded" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Upload failed (may already exist): $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "[2/7] Skipping data upload (local training or --SkipUpload)" -ForegroundColor Yellow
}
Write-Host ""

# Step 3: Train model
Write-Host "[3/7] Training model with robust augmentation..." -ForegroundColor Cyan
Write-Host "  This will take ~10-15 minutes on Kaggle T4 GPU" -ForegroundColor Gray
Write-Host "  Config: Enhanced brightness (±40%), quality degradation, background swap" -ForegroundColor Gray
Write-Host ""

if ($LocalOnly) {
    Write-Host "  ERROR: Local training not recommended (requires GPU)" -ForegroundColor Red
    Write-Host "  Remove -LocalOnly to train on Kaggle, or use CPU with caution" -ForegroundColor Yellow
    exit 1
}

try {
    uv run coffeeguard remote train -c $ConfigFile --seeds $Seed
    Write-Host "  ✓ Training complete" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Training failed: $_" -ForegroundColor Red
    Write-Host "  Check logs in runs/$RunName/train.log" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Step 4: Export model bundle
Write-Host "[4/7] Exporting ONNX bundle..." -ForegroundColor Cyan
try {
    uv run coffeeguard export --run "runs/$RunName" --name $BundleName
    Write-Host "  ✓ Bundle exported to artifacts/models/$BundleName" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Export failed: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Fit OOD gates with RELAXED thresholds
Write-Host "[5/7] Fitting OOD gates (relaxed for real-world robustness)..." -ForegroundColor Cyan
try {
    uv run coffeeguard ood collect
    uv run coffeeguard ood fit -b "artifacts/models/$BundleName"
    Write-Host "  ✓ OOD gates fitted" -ForegroundColor Green
} catch {
    Write-Host "  ✗ OOD fitting failed: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 6: Evaluate model
Write-Host "[6/7] Evaluating model performance..." -ForegroundColor Cyan
try {
    uv run coffeeguard evaluate -b "artifacts/models/$BundleName"
    Write-Host "  ✓ Evaluation complete" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Evaluation had warnings: $_" -ForegroundColor Yellow
}
Write-Host ""

# Step 7: Test robustness
Write-Host "[7/7] Testing robustness to corruptions..." -ForegroundColor Cyan
try {
    uv run coffeeguard robustness -b "artifacts/models/$BundleName"
    Write-Host "  ✓ Robustness testing complete" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Robustness testing had warnings: $_" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "=== Retraining Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "New bundle: artifacts/models/$BundleName" -ForegroundColor Cyan
Write-Host "Run folder: runs/$RunName" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review metrics in artifacts/eval/$BundleName/metrics.json"
Write-Host "2. Compare with baseline: artifacts/eval/cand-effv2b0-bgswap/"
Write-Host "3. Test with bright/PNG web images"
Write-Host "4. If satisfied, deploy:"
Write-Host "   - Update MODEL_BUNDLE env var or .env file"
Write-Host "   - Restart API: uv run uvicorn app.main:app --app-dir apps/api"
Write-Host ""
Write-Host "To deploy immediately:" -ForegroundColor Green
Write-Host "  Update bundle.json symlink or MODEL_BUNDLE to: $BundleName"
Write-Host ""
