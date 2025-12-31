# Build flex_gemm extension with full verbose output

$WORKDIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $WORKDIR

# Set up ROCm/HIP paths
$rocmPath = "C:\Program Files\AMD\ROCm\6.4"
$env:HIP_PATH = $rocmPath
$env:ROCM_PATH = $rocmPath
$env:PATH = "$rocmPath\bin;$env:PATH"
$env:GPU_ARCHS = "gfx1151"
$env:PYTORCH_ROCM_ARCH = "gfx1151"
$env:FORCE_CUDA = "1"

$venvPython = Join-Path $WORKDIR ".venv\Scripts\python.exe"

Write-Host "Building flex_gemm..."
Set-Location (Join-Path $WORKDIR "extensions\flex_gemm")

# Clean
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
Get-ChildItem -Filter "*.egg-info" | Remove-Item -Recurse -Force

# Build with verbose output saved to log
$logFile = Join-Path $WORKDIR "flex_gemm_build.log"
& $venvPython setup.py build_ext --inplace 2>&1 | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "Build exit code: $LASTEXITCODE"
Write-Host "Log saved to: $logFile"

# Show last 100 lines if failed
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Last 100 lines of output:" -ForegroundColor Yellow
    Get-Content $logFile | Select-Object -Last 100
}
