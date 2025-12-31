# TRELLIS.2 AMD Port - Implementation Complete

## Summary

This document summarizes the AMD HIP port of TRELLIS.2 from its original CUDA-only implementation.

**Status: CORE IMPLEMENTATION COMPLETE**

The TRELLIS.2 codebase has been modified to support AMD GPUs with ROCm/HIP. All major components have been adapted with fallback paths for AMD hardware.

---

## Changes Made

### 1. Installation Script (`install_amd.sh`)

Created a comprehensive AMD installation script that:
- Detects ROCm platform
- Installs PyTorch for ROCm 6.2+
- Builds AMD-compatible extensions (nvdiffrast-hip, torchsparse, o-voxel)
- Creates `run_amd.sh` with appropriate environment variables

### 2. Sparse Attention SDPA Fallback

**File:** `trellis2/modules/sparse/attention/full_attn.py`

Added support for `sdpa` and `naive` attention backends:
- Converts variable-length sparse tensors to padded dense format
- Uses PyTorch's `F.scaled_dot_product_attention`
- Converts back to sparse format
- Full support for all overload signatures (qkv, q+kv, q+k+v)

### 3. Sparse Config Auto-Detection

**File:** `trellis2/modules/sparse/config.py`

Added automatic AMD detection:
- Detects AMD HIP via `torch.version.hip`
- Defaults to `torchsparse` backend on AMD (FlexGEMM is CUDA-only)
- Defaults to `sdpa` attention backend on AMD
- Added `sdpa` and `naive` to valid attention backends

### 4. DINOv2/v3 CPU Fallback

**File:** `trellis2/modules/image_feature_extractor.py`

Added CPU fallback for DINOv2/v3 feature extractors:
- AMD HIP has issues with hipBLAS/Tensile in DINOv2's attention layers
- On AMD, models run on CPU and results are moved to GPU
- Controlled via `TRELLIS_DINO_CPU` environment variable

### 5. Pipeline AMD Support

**File:** `trellis2/pipelines/base.py`

Added AMD-specific features:
- `warmup()` method to pre-initialize torchsparse kernels
- `print_platform_info()` static method for debugging
- Automatic platform info printing when loading on AMD
- AMD detection flags on pipeline instances

### 6. Mesh Operations Fallbacks

**File:** `trellis2/representations/mesh/base.py`

Added fallbacks for CUDA-only mesh operations:
- `fill_holes()`: Falls back to trimesh on AMD
- `remove_faces()`: Pure PyTorch implementation for AMD
- `simplify()`: Falls back to trimesh's `simplify_quadric_decimation`
- `grid_sample_3d`: PyTorch native `F.grid_sample` fallback

### 7. OpenGL Mesh Renderer

**File:** `trellis2/renderers/mesh_renderer_gl.py`

Updated existing OpenGL renderer for TRELLIS.2:
- Uses moderngl for pure OpenGL rendering
- Bypasses nvdiffrast HIP issues at high resolution
- Updated to use TRELLIS.2's mesh types (Mesh, MeshWithVoxel, MeshWithPbrMaterial)
- Added `create_mesh_renderer()` factory function with auto-detection

**File:** `trellis2/renderers/__init__.py`

Added exports for:
- `MeshRendererGL`
- `create_mesh_renderer`

### 8. Extensions Copied from TRELLIS-AMD v1

**Directory:** `extensions/`

- `nvdiffrast-hip/` - HIP-ported rasterizer
- `torchsparse/` - AMD-compatible sparse convolutions
- `diff-gaussian-rasterization/` - HIP Gaussian splatting

---

## Environment Variables

| Variable | Default (AMD) | Description |
|----------|---------------|-------------|
| `ATTN_BACKEND` | `sdpa` | Attention backend (sdpa, naive, flash_attn, xformers) |
| `SPARSE_ATTN_BACKEND` | `sdpa` | Sparse attention backend |
| `SPARSE_CONV_BACKEND` | `torchsparse` | Sparse conv backend (torchsparse, flex_gemm) |
| `TRELLIS_DINO_CPU` | `1` | Run DINOv2/v3 on CPU (1=yes, 0=no) |
| `TRELLIS_MESH_BACKEND` | `gl` | Mesh renderer backend (gl, hip, auto) |
| `XFORMERS_DISABLED` | `1` | Disable xformers import |

---

## Testing

Run the AMD test script to verify the port:

```bash
source .venv/bin/activate
python test_amd.py
```

This tests:
1. PyTorch/HIP detection
2. Sparse module configuration
3. Sparse tensor creation
4. SDPA attention
5. DINOv2 CPU fallback
6. OpenGL mesh renderer
7. Pipeline loading

---

## Known Limitations

1. **Performance**: AMD path may be 2-3x slower than NVIDIA due to:
   - DINOv2 running on CPU
   - Sparse attention using dense padding
   - OpenGL renderer instead of HIP rasterization

2. **PBR Materials**: nvdiffrec (PBR material baking) has no HIP port. Full PBR pipeline unavailable on AMD.

3. **Resolution Limits**: nvdiffrast-hip has a 128px resolution limit in some operations. OpenGL fallback removes this limit.

4. **Mesh Simplification**: Trimesh fallback may produce slightly different results than cumesh.

---

## Usage

### Quick Start

```bash
# Install
./install_amd.sh

# Run with AMD settings
./run_amd.sh
```

### Python API

```python
import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'torchsparse'
os.environ['TRELLIS_DINO_CPU'] = '1'

from trellis2.pipelines import Trellis2ImageTo3DPipeline

# Load pipeline (will print AMD info)
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# Optional: warmup sparse kernels
pipeline.warmup()

# Run inference
from PIL import Image
image = Image.open("input.png")
meshes = pipeline.run(image)
```

---

## File Summary

| File | Changes |
|------|---------|
| `install_amd.sh` | NEW - AMD installation script |
| `run_amd.sh` | NEW - Generated by install script |
| `test_amd.py` | NEW - AMD test suite |
| `trellis2/modules/sparse/config.py` | Modified - AMD auto-detection |
| `trellis2/modules/sparse/attention/full_attn.py` | Modified - SDPA fallback |
| `trellis2/modules/image_feature_extractor.py` | Modified - CPU fallback |
| `trellis2/pipelines/base.py` | Modified - warmup, platform info |
| `trellis2/representations/mesh/base.py` | Modified - cumesh/flex_gemm fallbacks |
| `trellis2/renderers/mesh_renderer_gl.py` | Modified - TRELLIS.2 mesh types |
| `trellis2/renderers/__init__.py` | Modified - Added GL renderer exports |
| `extensions/nvdiffrast-hip/` | Copied from TRELLIS-AMD v1 |
| `extensions/torchsparse/` | Copied from TRELLIS-AMD v1 |
| `extensions/diff-gaussian-rasterization/` | Copied from TRELLIS-AMD v1 |

---

## Next Steps

1. **Test on actual AMD hardware** - Run `test_amd.py` and full inference
2. **Performance profiling** - Identify remaining bottlenecks
3. **Optimize DINOv2** - Investigate aotriton for GPU execution
4. **Port cumesh** - Native HIP port would improve mesh operations
5. **Port nvdiffrec** - Enable full PBR pipeline on AMD
