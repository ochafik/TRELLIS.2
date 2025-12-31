# TRELLIS.2 AMD Port - Build Log

## 2024-12-31: Initial Port

### Session Start
- Cloned TRELLIS.2 from microsoft/TRELLIS.2
- Goal: Port to AMD GPUs with ROCm/HIP

### Analysis Phase
- Spawned 4 subagents to analyze codebase
- Identified key blockers:
  - FlexGEMM: CUDA-only sparse convolutions
  - Attention: Only flash_attn/xformers backends
  - DINOv2: hipBLAS crashes on AMD
  - nvdiffrast: Needs HIP port

### Implementation Phase

#### 1. Sparse Config (`trellis2/modules/sparse/config.py`)
- Added AMD HIP detection
- Auto-defaults to `torchsparse` on AMD
- Auto-defaults to `sdpa` attention on AMD
- Added `sdpa` and `naive` to valid backends

#### 2. Sparse SDPA Attention (`trellis2/modules/sparse/attention/full_attn.py`)
- Implemented full SDPA fallback
- Converts VarLenTensor to padded dense
- Uses PyTorch's F.scaled_dot_product_attention
- Converts back to sparse format
- Supports all overload signatures

#### 3. DINOv2/v3 CPU Fallback (`trellis2/modules/image_feature_extractor.py`)
- Added `_IS_AMD` and `_DINO_USE_CPU` flags
- DinoV2FeatureExtractor: CPU execution path
- DinoV3FeatureExtractor: CPU execution path
- Results moved back to GPU after extraction

#### 4. Pipeline AMD Support (`trellis2/pipelines/base.py`)
- Added `warmup()` method for torchsparse kernels
- Added `print_platform_info()` static method
- Auto-prints AMD info on model load
- Added `_is_amd` and `_warmup_done` flags

#### 5. Mesh Operations Fallbacks (`trellis2/representations/mesh/base.py`)
- Lazy imports for cumesh and flex_gemm
- `fill_holes()`: Falls back to trimesh
- `remove_faces()`: Pure PyTorch implementation
- `simplify()`: Falls back to trimesh
- `grid_sample_3d`: PyTorch native F.grid_sample

#### 6. OpenGL Mesh Renderer (`trellis2/renderers/mesh_renderer_gl.py`)
- Updated imports for TRELLIS.2 mesh types
- Added face normal computation fallback
- Added `transformation` parameter support
- Factory function with AMD auto-detection

#### 7. Scripts
- `install_amd.ps1`: Windows installation script
- `run_amd.ps1`: Run script with AMD env vars
- `test_amd.py`: 7-test AMD compatibility suite

### Test Results
All 7 tests pass:
```
PyTorch/HIP: PASS
Sparse Config: PASS
Sparse Tensor: PASS
SDPA Attention: PASS
DINOv2 CPU Fallback: PASS
OpenGL Renderer: PASS
Pipeline: PASS
```

### Extension Build Attempts

#### o-voxel
```
pip install . --no-build-isolation -v
```
- Hipify step: SUCCESS (13 kernel launches)
- HIP compilation: FAILED
- Error: Need to capture

#### cumesh
- Hipify step: SUCCESS (102 kernel launches)
- HIP compilation: FAILED
- Error: Need to capture

#### flex_gemm
- Hipify step: SUCCESS (15 kernel launches)
- HIP compilation: FAILED
- Error: Need to capture

### Next Steps
1. Capture detailed build errors for each extension
2. Fix compilation issues
3. Test full inference

---

## Build Error Investigation

### o-voxel Build Attempt 1
*Timestamp: TBD*
*Command: TBD*
*Error: TBD*

---
