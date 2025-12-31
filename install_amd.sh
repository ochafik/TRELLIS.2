#!/bin/bash
# TRELLIS.2 AMD Installation Script
# For AMD GPUs with ROCm 6.2+

set -e

echo "=========================================="
echo "TRELLIS.2 AMD Installation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect platform
if command -v rocminfo > /dev/null 2>&1; then
    PLATFORM="hip"
    echo -e "${GREEN}[OK] Detected AMD ROCm platform${NC}"
    rocminfo | grep "Name:" | head -1
elif command -v nvidia-smi > /dev/null 2>&1; then
    PLATFORM="cuda"
    echo -e "${YELLOW}[WARN] Detected NVIDIA platform - this script is for AMD GPUs${NC}"
    echo "Use setup.sh for NVIDIA installation"
    exit 1
else
    echo -e "${RED}[ERROR] No GPU platform detected${NC}"
    echo "Please install ROCm first: https://rocm.docs.amd.com/"
    exit 1
fi

WORKDIR=$(pwd)

# Check for required system packages
echo ""
echo "[1/8] Checking system dependencies..."
if command -v apt > /dev/null 2>&1; then
    # Debian/Ubuntu
    MISSING_PKGS=""
    for pkg in python3-venv python3-dev libsparsehash-dev libjpeg-dev; do
        if ! dpkg -l | grep -q "^ii  $pkg"; then
            MISSING_PKGS="$MISSING_PKGS $pkg"
        fi
    done
    if [ -n "$MISSING_PKGS" ]; then
        echo -e "${YELLOW}Installing missing packages:$MISSING_PKGS${NC}"
        sudo apt update && sudo apt install -y $MISSING_PKGS
    fi
fi

# Create virtual environment
echo ""
echo "[2/8] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}[OK] Created .venv${NC}"
else
    echo -e "${YELLOW}[SKIP] .venv already exists${NC}"
fi

source .venv/bin/activate
echo "Python: $(python --version)"

# Install PyTorch for ROCm
echo ""
echo "[3/8] Installing PyTorch for ROCm..."
pip install --upgrade pip wheel setuptools

# Check if torch is already installed with ROCm
if python -c "import torch; assert hasattr(torch.version, 'hip') and torch.version.hip is not None" 2>/dev/null; then
    echo -e "${YELLOW}[SKIP] PyTorch ROCm already installed${NC}"
else
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    echo -e "${GREEN}[OK] PyTorch ROCm installed${NC}"
fi

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, HIP: {torch.version.hip}, CUDA available: {torch.cuda.is_available()}')"

# Install basic dependencies
echo ""
echo "[4/8] Installing Python dependencies..."
pip install ninja  # Build tool first
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless
pip install trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard
pip install kornia timm safetensors plyfile moderngl pillow
pip install huggingface_hub  # For model downloading

# Install utils3d (specific commit)
echo "Installing utils3d..."
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Build AMD extensions
echo ""
echo "[5/8] Building nvdiffrast-hip..."
if [ -d "extensions/nvdiffrast-hip" ]; then
    cd extensions/nvdiffrast-hip
    pip install . --no-build-isolation -v 2>&1 | tail -20
    cd "$WORKDIR"
    echo -e "${GREEN}[OK] nvdiffrast-hip built${NC}"
else
    echo -e "${YELLOW}[SKIP] nvdiffrast-hip not found in extensions/${NC}"
fi

echo ""
echo "[6/8] Building torchsparse..."
if [ -d "extensions/torchsparse" ]; then
    cd extensions/torchsparse
    FORCE_CUDA=1 pip install . --no-build-isolation -v 2>&1 | tail -20
    cd "$WORKDIR"
    echo -e "${GREEN}[OK] torchsparse built${NC}"
else
    echo -e "${YELLOW}[SKIP] torchsparse not found in extensions/${NC}"
fi

echo ""
echo "[7/8] Building o-voxel..."
if [ -d "o-voxel" ]; then
    cd o-voxel
    pip install . --no-build-isolation -v 2>&1 | tail -20
    cd "$WORKDIR"
    echo -e "${GREEN}[OK] o-voxel built${NC}"
else
    echo -e "${RED}[ERROR] o-voxel directory not found${NC}"
fi

# Optional: Build diff-gaussian-rasterization
echo ""
echo "[8/8] Building diff-gaussian-rasterization (optional)..."
if [ -d "extensions/diff-gaussian-rasterization" ]; then
    cd extensions/diff-gaussian-rasterization
    pip install . --no-build-isolation -v 2>&1 | tail -20 || echo -e "${YELLOW}[WARN] diff-gaussian-rasterization build failed (optional)${NC}"
    cd "$WORKDIR"
else
    echo -e "${YELLOW}[SKIP] diff-gaussian-rasterization not found${NC}"
fi

# Create run script
echo ""
echo "Creating run_amd.sh..."
cat > run_amd.sh << 'EOF'
#!/bin/bash
# TRELLIS.2 AMD Run Script

# Activate environment
source .venv/bin/activate

# AMD-specific environment variables
export ATTN_BACKEND=sdpa                    # Use PyTorch SDPA (most compatible)
export SPARSE_ATTN_BACKEND=sdpa             # Sparse attention fallback
export SPARSE_CONV_BACKEND=torchsparse      # Use torchsparse instead of flex_gemm

# Rendering settings
export TRELLIS_AMD_RASTER_RES=128           # Limit nvdiffrast resolution
export TRELLIS_MESH_BACKEND=gl              # Use pure OpenGL renderer
export TRELLIS_RASTER_BACKEND=gl            # Use OpenGL for fill_holes

# Texture baking
export TRELLIS_TEXTURE_MODE=fast            # Fast mode for AMD
export TRELLIS_TEXTURE_SIZE=1024            # Texture resolution

# Disable incompatible libraries
export XFORMERS_DISABLED=1

# Memory optimization
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Run the application
python app.py "$@"
EOF
chmod +x run_amd.sh

echo ""
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "To run TRELLIS.2 on AMD:"
echo "  ./run_amd.sh"
echo ""
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  export ATTN_BACKEND=sdpa"
echo "  export SPARSE_CONV_BACKEND=torchsparse"
echo "  python app.py"
echo ""
