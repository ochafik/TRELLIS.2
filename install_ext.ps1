# TRELLIS.2 Install All Extensions for AMD HIP
# Installs flex_gemm, cumesh, and o-voxel in correct order

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

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TRELLIS.2 Install All Extensions (AMD HIP)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

function Install-Extension {
    param([string]$Name, [string]$Path)

    Write-Host ""
    Write-Host "Installing: $Name" -ForegroundColor Yellow
    Write-Host "Path: $Path"

    if (-not (Test-Path $Path)) {
        Write-Host "[$Name] ERROR: Path not found" -ForegroundColor Red
        return $false
    }

    Push-Location $Path

    # Clean build artifacts
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    Get-ChildItem -Filter "*.egg-info" | Remove-Item -Recurse -Force

    Write-Host "Running pip install..."
    & $venvPython -m pip install . --no-build-isolation --no-deps -v 2>&1 | Select-Object -Last 30

    $result = $LASTEXITCODE
    Pop-Location

    if ($result -eq 0) {
        Write-Host "[$Name] Install SUCCESS" -ForegroundColor Green
        return $true
    } else {
        Write-Host "[$Name] Install FAILED" -ForegroundColor Red
        return $false
    }
}

$results = @{}

# Install in order of dependencies
$results["flex_gemm"] = Install-Extension "flex_gemm" (Join-Path $WORKDIR "extensions\flex_gemm")
$results["cumesh"] = Install-Extension "cumesh" (Join-Path $WORKDIR "extensions\cumesh")
$results["o-voxel"] = Install-Extension "o-voxel" (Join-Path $WORKDIR "o-voxel")

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Install Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
foreach ($ext in @("flex_gemm", "cumesh", "o-voxel")) {
    $status = if ($results[$ext]) { "PASS" } else { "FAIL" }
    $color = if ($results[$ext]) { "Green" } else { "Red" }
    Write-Host "  $ext : $status" -ForegroundColor $color
}

# Test imports
Write-Host ""
Write-Host "Testing imports..." -ForegroundColor Yellow
& $venvPython (Join-Path $WORKDIR "check_ext.py")
