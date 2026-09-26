# Quick robustness test for bright/high-res/PNG images
# Tests the current model against synthetic brightness variations
# without requiring actual web-downloaded images

Write-Host "=== Quick Robustness Test ===" -ForegroundColor Cyan
Write-Host ""

# Test if API is running
Write-Host "Checking API status..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
    Write-Host "✓ API running: $($health.model)" -ForegroundColor Green
} catch {
    Write-Host "✗ API not running. Start with:" -ForegroundColor Red
    Write-Host "  uv run uvicorn app.main:app --app-dir apps/api --host 0.0.0.0 --port 8000" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Testing with sample images + brightness variations..." -ForegroundColor Cyan
Write-Host ""

# Use Python to test with PIL brightness adjustments
$TestScript = @"
import sys
from pathlib import Path
from PIL import Image, ImageEnhance
import httpx
import io

samples_dir = Path('apps/web/samples')
if not samples_dir.exists():
    print('No samples directory found')
    sys.exit(1)

# Test each sample with brightness variations
results = {'accepted': 0, 'rejected': 0, 'uncertain': 0}
brightness_factors = [0.7, 1.0, 1.3, 1.6, 2.0]  # dark to bright

for sample in samples_dir.glob('*.jpg'):
    if 'not_coffee' in sample.name:
        continue
    
    print(f'\nTesting: {sample.name}')
    img = Image.open(sample)
    
    for factor in brightness_factors:
        # Apply brightness adjustment
        enhancer = ImageEnhance.Brightness(img)
        img_adj = enhancer.enhance(factor)
        
        # Convert to bytes
        buf = io.BytesIO()
        img_adj.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        
        # Send to API
        try:
            r = httpx.post(
                'http://localhost:8000/predict',
                files={'file': (f'{sample.stem}_br{factor}.jpg', buf, 'image/jpeg')},
                timeout=10
            )
            r.raise_for_status()
            result = r.json()
            status = result['decision']
            results[status] = results.get(status, 0) + 1
            
            # Show results for extreme brightness
            if factor in [0.7, 2.0]:
                conf = result.get('confidence', 0)
                pred = result.get('predicted_class', 'N/A')
                print(f'  Brightness {factor:0.1f}x: {status:10s} | {pred:12s} (conf {conf:.2f})')
        except Exception as e:
            print(f'  Brightness {factor:0.1f}x: ERROR - {e}')
            results['error'] = results.get('error', 0) + 1

print('\n=== Summary ===')
total = sum(results.values())
print(f'Total tests: {total}')
for status, count in results.items():
    pct = 100 * count / total if total > 0 else 0
    print(f'  {status:10s}: {count:3d} ({pct:5.1f}%)')

# Good if >80% accepted
acceptance_rate = 100 * results['accepted'] / total if total > 0 else 0
if acceptance_rate >= 80:
    print(f'\n✓ GOOD: {acceptance_rate:.1f}% acceptance rate')
    sys.exit(0)
else:
    print(f'\n⚠ LOW: {acceptance_rate:.1f}% acceptance rate (target >80%)')
    sys.exit(1)
"@

uv run python -c $TestScript

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Robustness test passed!" -ForegroundColor Green
} else {
    Write-Host "`n⚠ Robustness test shows issues" -ForegroundColor Yellow
    Write-Host "Consider retraining with robust config:" -ForegroundColor Yellow
    Write-Host "  .\scripts\retrain_robust.ps1" -ForegroundColor White
}
