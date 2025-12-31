# Build cumesh extension for AMD HIP
$WORKDIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$WORKDIR\extensions\cumesh"

# Set up ROCm/HIP paths
$rocmPath = "C:\Program Files\AMD\ROCm\6.4"
$env:HIP_PATH = $rocmPath
$env:ROCM_PATH = $rocmPath
$env:PATH = "$rocmPath\bin;$env:PATH"
$env:GPU_ARCHS = "gfx1151"
$env:PYTORCH_ROCM_ARCH = "gfx1151"
$env:FORCE_CUDA = "1"

$venvPython = Join-Path $WORKDIR ".venv\Scripts\python.exe"

# Clean build artifacts
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
Get-ChildItem -Filter "*.egg-info" | Remove-Item -Recurse -Force

Write-Host "Building cumesh HIP extension..." -ForegroundColor Yellow
& $venvPython setup.py build_ext --inplace 2>&1

Write-Host ""
Write-Host "Build exit code: $LASTEXITCODE" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
