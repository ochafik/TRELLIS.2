# TRELLIS.2 AMD Porting Quick Start

## Prerequisites

- AMD GPU (RX 7000/9000 series or Ryzen AI APU)
- ROCm 6.2+ (Linux) or AMD HIP SDK (Windows experimental)
- Python 3.10+
- 24GB+ VRAM recommended

## Step 1: Copy Extensions from TRELLIS-AMD v1

```bash
# Clone both repos
cd /path/to/workspace
git clone https://github.com/microsoft/TRELLIS.2.git TRELLIS.2-AMD
git clone https://github.com/YourFork/TRELLIS-AMD.git  # Your v1 AMD fork

# Create extensions directory
mkdir -p TRELLIS.2-AMD/extensions

# Copy AMD-compatible extensions
cp -r TRELLIS-AMD/extensions/nvdiffrast-hip TRELLIS.2-AMD/extensions/
cp -r TRELLIS-AMD/extensions/torchsparse TRELLIS.2-AMD/extensions/
cp -r TRELLIS-AMD/extensions/diff-gaussian-rasterization TRELLIS.2-AMD/extensions/
```

## Step 2: Create AMD Installation Script

Create `install_amd.sh`:

```bash
#!/bin/bash
set -e

# Detect platform
if command -v rocminfo > /dev/null; then
    PLATFORM="hip"
    echo "Detected AMD ROCm platform"
else
    echo "Error: ROCm not found. Please install ROCm first."
    exit 1
fi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch for ROCm
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4

# Install basic dependencies
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja
pip install trimesh transformers gradio tensorboard pandas lpips zstandard
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install kornia timm safetensors plyfile moderngl

# Build AMD extensions
echo "Building nvdiffrast-hip..."
cd extensions/nvdiffrast-hip
pip install . --no-build-isolation
cd ../..

echo "Building torchsparse..."
cd extensions/torchsparse
FORCE_CUDA=1 pip install . --no-build-isolation
cd ../..

echo "Building o-voxel..."
cd o-voxel
pip install . --no-build-isolation
cd ..

echo "Installation complete!"
```

## Step 3: Set Environment Variables

Create `run_amd.sh`:

```bash
#!/bin/bash

# Attention backend - use PyTorch SDPA (most compatible)
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa  # Requires implementation

# Sparse convolution - use torchsparse instead of flex_gemm
export SPARSE_CONV_BACKEND=torchsparse

# Rendering - limit resolution for nvdiffrast stability
export TRELLIS_AMD_RASTER_RES=128
export TRELLIS_MESH_BACKEND=gl

# Texture baking
export TRELLIS_TEXTURE_MODE=fast
export TRELLIS_TEXTURE_SIZE=1024

# Disable xformers (not fully AMD compatible)
export XFORMERS_DISABLED=1

# Run the app
source .venv/bin/activate
python app.py
```

## Step 4: Required Code Modifications

### 4.1 Add DINOv2 CPU Fallback

Edit `trellis2/modules/image_feature_extractor.py`:

```python
class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self._is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None
        # ... rest of init

    def forward(self, images):
        if self._is_amd:
            # Move to CPU to avoid hipBLAS issues
            device = images.device
            self.model.to('cpu')
            images_cpu = images.to('cpu')
            with torch.no_grad():
                features = self.model(images_cpu)
            features = {k: v.to(device) for k, v in features.items()}
            self.model.to(device)  # Move back for potential other uses
            return features
        else:
            return self.model(images)
```

### 4.2 Add Sparse SDPA Fallback

Edit `trellis2/modules/sparse/attention/full_attn.py`:

```python
def sparse_scaled_dot_product_attention(qkv, ...):
    if ATTN_BACKEND in ['xformers', 'flash_attn', 'flash_attn_3']:
        # Existing implementation
        ...
    elif ATTN_BACKEND == 'sdpa':
        # SDPA fallback for AMD
        q, k, v = qkv.chunk(3, dim=-1)
        # Unserialize for standard attention
        q_dense = varlen_to_dense(q, ...)  # Need to implement
        k_dense = varlen_to_dense(k, ...)
        v_dense = varlen_to_dense(v, ...)
        out = F.scaled_dot_product_attention(q_dense, k_dense, v_dense)
        return dense_to_varlen(out, ...)  # Need to implement
    else:
        raise NotImplementedError(f"Sparse attention with {ATTN_BACKEND}")
```

### 4.3 Add Torchsparse Backend Check

Edit `trellis2/modules/sparse/config.py`:

```python
import os
import torch

SPARSE_CONV_BACKEND = os.environ.get('SPARSE_CONV_BACKEND', 'auto').lower()

# Auto-detect best backend
if SPARSE_CONV_BACKEND == 'auto':
    _is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None
    if _is_amd:
        SPARSE_CONV_BACKEND = 'torchsparse'  # Use torchsparse on AMD
    else:
        SPARSE_CONV_BACKEND = 'flex_gemm'    # Use flex_gemm on NVIDIA
```

## Step 5: Test Basic Inference

```python
import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'torchsparse'

import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline

# Check AMD detection
print(f"AMD GPU: {hasattr(torch.version, 'hip') and torch.version.hip is not None}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Load pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# Test with sample image
from PIL import Image
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
```

## Known Issues & Workarounds

### Issue 1: "unspecified launch failure" during sparse conv
**Cause:** Torchsparse needs warmup on AMD
**Fix:** Add warmup call before first inference

### Issue 2: Mesh rendering shows artifacts
**Cause:** nvdiffrast multi-bin race condition
**Fix:** Set `TRELLIS_AMD_RASTER_RES=128` or use GL backend

### Issue 3: DINOv2 returns NaN/zero
**Cause:** hipBLAS kernel issues with certain matrix sizes
**Fix:** Run DINOv2 on CPU (see Step 4.1)

### Issue 4: Out of memory at 1024³ resolution
**Cause:** AMD memory management differs from NVIDIA
**Fix:** Use `pipeline.run(..., sparse_structure_sampler_params={'resolution': 512})`

## Performance Expectations

| Resolution | NVIDIA H100 | AMD RX 7900 XTX (est.) |
|------------|-------------|------------------------|
| 512³ | ~3s | ~10-15s |
| 1024³ | ~17s | ~45-60s |
| 1536³ | ~60s | ~3-5min |

Note: AMD performance is estimated based on TRELLIS v1 benchmarks. Actual performance will depend on optimizations and hardware.

## Next Steps

1. Run basic inference test
2. If successful, test GLB export
3. Report issues to create targeted fixes
4. Optimize hot paths identified in profiling
