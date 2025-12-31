# TRELLIS.2 Build All Extensions for AMD HIP
# Builds flex_gemm, cumesh, and o-voxel in correct order

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
Write-Host "TRELLIS.2 Build All Extensions (AMD HIP)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "HIP_PATH: $env:HIP_PATH"
Write-Host "GPU_ARCHS: $env:GPU_ARCHS"
Write-Host "hipcc:" (& hipcc --version 2>&1 | Select-Object -First 1)
Write-Host ""

function Build-Extension {
    param([string]$Name, [string]$Path)

    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host "Building: $Name" -ForegroundColor Yellow
    Write-Host "Path: $Path"
    Write-Host "==========================================" -ForegroundColor Yellow

    if (-not (Test-Path $Path)) {
        Write-Host "[$Name] ERROR: Path not found" -ForegroundColor Red
        return $false
    }

    Push-Location $Path

    # Clean build artifacts
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    Get-ChildItem -Filter "*.egg-info" | Remove-Item -Recurse -Force

    Write-Host "Installing $Name..."
    & $venvPython -m pip install . --no-build-isolation --no-deps -v 2>&1 | Tee-Object -Variable buildOutput

    $result = $LASTEXITCODE
    Pop-Location

    if ($result -eq 0) {
        Write-Host "[$Name] Build SUCCESS" -ForegroundColor Green
        return $true
    } else {
        Write-Host "[$Name] Build FAILED" -ForegroundColor Red
        # Show last 50 lines of build output for debugging
        $buildOutput | Select-Object -Last 50 | ForEach-Object { Write-Host $_ }
        return $false
    }
}

$results = @{}

# Build in order of dependencies
# 1. flex_gemm (no dependencies)
$results["flex_gemm"] = Build-Extension "flex_gemm" (Join-Path $WORKDIR "extensions\flex_gemm")

# 2. cumesh (no dependencies)
$results["cumesh"] = Build-Extension "cumesh" (Join-Path $WORKDIR "extensions\cumesh")

# 3. o-voxel (depends on flex_gemm and cumesh)
$results["o-voxel"] = Build-Extension "o-voxel" (Join-Path $WORKDIR "o-voxel")

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Build Summary" -ForegroundColor Cyan
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
