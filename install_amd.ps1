# TRELLIS.2 AMD Installation Script for Windows
# For AMD GPUs with ROCm/HIP SDK

# Don't stop on every error - we'll handle errors explicitly
$ErrorActionPreference = "Continue"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TRELLIS.2 AMD Installation (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Get script directory
$WORKDIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $WORKDIR

# Set up ROCm/HIP paths
$rocmPath = "C:\Program Files\AMD\ROCm\6.4"
if (Test-Path $rocmPath) {
    Write-Host "[ROCm] Found ROCm 6.4 at $rocmPath" -ForegroundColor Green
    $env:HIP_PATH = $rocmPath
    $env:ROCM_PATH = $rocmPath
    $env:PATH = "$rocmPath\bin;$env:PATH"
    $env:CMAKE_PREFIX_PATH = "$rocmPath;$env:CMAKE_PREFIX_PATH"
} else {
    Write-Host "[WARN] ROCm 6.4 not found at $rocmPath - HIP extensions may fail to build" -ForegroundColor Yellow
}

# Check for Python
Write-Host ""
Write-Host "[1/8] Checking Python..." -ForegroundColor Yellow
$pythonCmd = $null
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.(1[0-9]|[0-9])") {
        $pythonCmd = "python"
        Write-Host "[OK] Found $pythonVersion" -ForegroundColor Green
    }
}
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python 3.10+ not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[2/8] Creating Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    & $pythonCmd -m venv .venv
    Write-Host "[OK] Created .venv" -ForegroundColor Green
} else {
    Write-Host "[SKIP] .venv already exists" -ForegroundColor Yellow
}

# Activate venv
$venvPython = Join-Path $WORKDIR ".venv\Scripts\python.exe"
$venvPip = Join-Path $WORKDIR ".venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment not properly created" -ForegroundColor Red
    exit 1
}

Write-Host "Using: $venvPython"

# Install PyTorch for ROCm
Write-Host ""
Write-Host "[3/8] Installing PyTorch for ROCm..." -ForegroundColor Yellow

# Check if torch is already installed with ROCm
$torchInstalled = $false
try {
    $torchCheck = & $venvPython -c "import torch; print(hasattr(torch.version, 'hip') and torch.version.hip is not None)" 2>$null
    if ($torchCheck -eq "True") {
        $torchInstalled = $true
    }
} catch {
    $torchInstalled = $false
}

if ($torchInstalled) {
    Write-Host "[SKIP] PyTorch ROCm already installed" -ForegroundColor Yellow
} else {
    & $venvPip install --upgrade pip wheel setuptools
    & $venvPip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] PyTorch installation failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] PyTorch ROCm installed" -ForegroundColor Green
}

# Verify PyTorch
& $venvPython -c "import torch; print(f'PyTorch {torch.__version__}, HIP: {torch.version.hip}, CUDA available: {torch.cuda.is_available()}')"

# Install basic dependencies
Write-Host ""
Write-Host "[4/8] Installing Python dependencies..." -ForegroundColor Yellow
& $venvPip install ninja
& $venvPip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless
& $venvPip install trimesh transformers "gradio==6.0.1" tensorboard pandas lpips zstandard
& $venvPip install kornia timm safetensors plyfile moderngl pillow
& $venvPip install huggingface_hub

# Install utils3d
Write-Host "Installing utils3d..."
& $venvPip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Build AMD extensions
Write-Host ""
Write-Host "[5/8] Building nvdiffrast-hip..." -ForegroundColor Yellow
$nvdiffrastPath = Join-Path $WORKDIR "extensions\nvdiffrast-hip"
if (Test-Path $nvdiffrastPath) {
    Push-Location $nvdiffrastPath
    & $venvPip install . --no-build-isolation -v 2>&1 | Select-Object -Last 20
    Pop-Location
    Write-Host "[OK] nvdiffrast-hip built" -ForegroundColor Green
} else {
    Write-Host "[SKIP] nvdiffrast-hip not found in extensions/" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[6/8] Building torchsparse..." -ForegroundColor Yellow
$torchsparsePath = Join-Path $WORKDIR "extensions\torchsparse"
if (Test-Path $torchsparsePath) {
    Push-Location $torchsparsePath
    $env:FORCE_CUDA = "1"
    & $venvPip install . --no-build-isolation -v 2>&1 | Select-Object -Last 20
    Pop-Location
    Write-Host "[OK] torchsparse built" -ForegroundColor Green
} else {
    Write-Host "[SKIP] torchsparse not found in extensions/" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[7/8] Building o-voxel..." -ForegroundColor Yellow
$ovoxelPath = Join-Path $WORKDIR "o-voxel"
if (Test-Path $ovoxelPath) {
    Push-Location $ovoxelPath
    & $venvPip install . --no-build-isolation -v 2>&1 | Select-Object -Last 20
    Pop-Location
    Write-Host "[OK] o-voxel built" -ForegroundColor Green
} else {
    Write-Host "[ERROR] o-voxel directory not found" -ForegroundColor Red
}

# Optional: Build diff-gaussian-rasterization
Write-Host ""
Write-Host "[8/8] Building diff-gaussian-rasterization (optional)..." -ForegroundColor Yellow
$gaussianPath = Join-Path $WORKDIR "extensions\diff-gaussian-rasterization"
if (Test-Path $gaussianPath) {
    Push-Location $gaussianPath
    try {
        & $venvPip install . --no-build-isolation -v 2>&1 | Select-Object -Last 20
        Write-Host "[OK] diff-gaussian-rasterization built" -ForegroundColor Green
    } catch {
        Write-Host "[WARN] diff-gaussian-rasterization build failed (optional)" -ForegroundColor Yellow
    }
    Pop-Location
} else {
    Write-Host "[SKIP] diff-gaussian-rasterization not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run TRELLIS.2 on AMD:" -ForegroundColor White
Write-Host "  .\run_amd.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or manually:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  `$env:ATTN_BACKEND='sdpa'" -ForegroundColor Yellow
Write-Host "  `$env:SPARSE_CONV_BACKEND='torchsparse'" -ForegroundColor Yellow
Write-Host "  python app.py" -ForegroundColor Yellow
Write-Host ""
