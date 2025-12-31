# TRELLIS.2 AMD Run Script for Windows
# Sets AMD-specific environment variables and runs the app

param(
    [string]$Script = "app.py",
    [switch]$Test,
    [switch]$Help
)

if ($Help) {
    Write-Host "TRELLIS.2 AMD Run Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run_amd.ps1              # Run app.py"
    Write-Host "  .\run_amd.ps1 -Test        # Run AMD test suite"
    Write-Host "  .\run_amd.ps1 -Script X.py # Run custom script"
    Write-Host ""
    exit 0
}

# Get script directory
$WORKDIR = $PSScriptRoot
if (-not $WORKDIR) {
    $WORKDIR = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location $WORKDIR

# Check venv exists
$venvPython = Join-Path $WORKDIR ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment not found at $venvPython" -ForegroundColor Red
    Write-Host "Run install_amd.ps1 first or copy .venv from TRELLIS-AMD" -ForegroundColor Yellow
    exit 1
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TRELLIS.2 AMD Runner (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# AMD-specific environment variables
Write-Host ""
Write-Host "Setting AMD environment variables..." -ForegroundColor Yellow

# Attention backend - use PyTorch SDPA (most compatible)
$env:ATTN_BACKEND = "sdpa"
$env:SPARSE_ATTN_BACKEND = "sdpa"

# Sparse convolution - use torchsparse instead of flex_gemm
$env:SPARSE_CONV_BACKEND = "torchsparse"

# DINOv2 - run on CPU to avoid hipBLAS issues
$env:TRELLIS_DINO_CPU = "1"

# Rendering settings
$env:TRELLIS_AMD_RASTER_RES = "128"
$env:TRELLIS_MESH_BACKEND = "gl"
$env:TRELLIS_RASTER_BACKEND = "gl"

# Texture baking
$env:TRELLIS_TEXTURE_MODE = "fast"
$env:TRELLIS_TEXTURE_SIZE = "1024"

# Disable incompatible libraries
$env:XFORMERS_DISABLED = "1"

# Memory optimization
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"

# Enable experimental aotriton for better SDPA performance
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"

Write-Host "  ATTN_BACKEND = $env:ATTN_BACKEND"
Write-Host "  SPARSE_CONV_BACKEND = $env:SPARSE_CONV_BACKEND"
Write-Host "  TRELLIS_DINO_CPU = $env:TRELLIS_DINO_CPU"
Write-Host "  TRELLIS_MESH_BACKEND = $env:TRELLIS_MESH_BACKEND"

# Determine what to run
if ($Test) {
    $Script = "test_amd.py"
}

Write-Host ""
Write-Host "Running: $Script" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Run the application
& $venvPython $Script @args
