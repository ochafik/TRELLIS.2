# TRELLIS.2 AMD Portability Assessment

## Executive Summary

TRELLIS.2 is a 4B parameter 3D generative model that requires significant porting effort for AMD GPUs. This document provides a comprehensive analysis of the codebase, identifies blockers, and outlines the porting strategy.

**Overall Assessment: MODERATE-HIGH EFFORT**

| Component | Effort | Risk | Status |
|-----------|--------|------|--------|
| Core PyTorch/Transformers | Low | Low | Compatible via ROCm |
| Attention Mechanisms | Medium | Medium | Needs fallback to SDPA |
| Sparse Convolutions (FlexGEMM) | High | High | No HIP support, needs torchsparse |
| O-Voxel Extension | Medium | Low | Has HIP awareness in setup.py |
| Rendering (nvdiffrast) | High | High | Needs nvdiffrast-hip from v1 |
| Mesh Operations (CuMesh) | Medium | Medium | Unknown HIP support |
| PBR Materials (nvdiffrec) | High | High | No HIP support exists |

---

## 1. Architecture Overview

### 1.1 Pipeline Flow

```
Image Input
    │
    ▼
┌─────────────────────────────────────┐
│  Image Feature Extraction (DINOv2)  │  ← AMD Issue: hipBLAS/Tensile bugs
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Sparse Structure Flow (512³)       │  ← Uses FlexGEMM (CUDA-only)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Shape Latent Flow (512→1024→1536)  │  ← Cascade sampling
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Texture Latent Flow                │  ← PBR material generation
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Decode to Mesh (O-Voxel → Mesh)    │  ← O-Voxel CUDA extension
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  PBR Rendering (nvdiffrast/rec)     │  ← CRITICAL: No HIP support
└─────────────────────────────────────┘
    │
    ▼
Output: GLB with PBR Textures
```

### 1.2 Model Components

| Component | Parameters | Backend | AMD Status |
|-----------|------------|---------|------------|
| DINOv2/v3 Feature Extractor | ~300M | PyTorch | CPU fallback needed |
| Sparse Structure VAE | ~200M | FlexGEMM | Needs torchsparse |
| Shape Flow Model | ~1.5B | Flash Attention | SDPA fallback |
| Texture Flow Model | ~1.5B | Flash Attention | SDPA fallback |
| Shape Decoder | ~200M | FlexGEMM | Needs torchsparse |
| Texture Decoder | ~200M | FlexGEMM | Needs torchsparse |

---

## 2. CUDA Extensions Inventory

### 2.1 O-Voxel (Included in Repo)

**Location:** `o-voxel/`

**Purpose:** Sparse voxel representation, mesh conversion, rasterization

**Source Files:**
- `src/hash/hash.cu` - GPU hashmap operations
- `src/rasterize/rasterize.cu` - Voxel rasterization
- `src/serialize/api.cu, z_order.cu, hilbert.cu` - Spatial encoding

**CUDA Features Used:**
- `cooperative_groups` - Block-level coordination
- `atomicCAS` - Thread-safe insertion
- `__syncthreads()` - Synchronization barriers
- `__launch_bounds__()` - Kernel configuration

**HIP Status:** ✅ SUPPORTED - setup.py has HIP detection and `--offload-arch` flags

```python
# From o-voxel/setup.py
IS_HIP = hasattr(torch.version, 'hip') and torch.version.hip is not None
if IS_HIP:
    # Uses --offload-arch for AMD GPU targets
```

**Porting Effort:** LOW - Already HIP-aware

---

### 2.2 FlexGEMM (External)

**Repository:** https://github.com/JeffreyXiang/FlexGEMM

**Purpose:** High-performance sparse convolutions using Triton

**AMD Status:** ❌ NOT SUPPORTED - Triton for AMD is experimental

**Porting Strategy:** Replace with torchsparse backend (from TRELLIS-AMD v1)

**Configuration:**
```python
# trellis2/modules/sparse/config.py
SPARSE_CONV_BACKEND = os.environ.get('SPARSE_CONV_BACKEND', 'flex_gemm')
# Options: 'none', 'spconv', 'torchsparse', 'flex_gemm'
```

**Porting Effort:** MEDIUM - Need to add/verify torchsparse backend

---

### 2.3 CuMesh (External)

**Repository:** https://github.com/JeffreyXiang/CuMesh

**Purpose:** GPU mesh operations (simplification, remeshing, UV unwrapping)

**AMD Status:** ❓ UNKNOWN - Need to investigate

**Used For:**
- Mesh simplification
- Hole filling
- UV parametrization

**Fallback Option:** Trimesh (CPU) for mesh operations

**Porting Effort:** MEDIUM - May need CPU fallback

---

### 2.4 nvdiffrast (External)

**Original:** https://github.com/NVlabs/nvdiffrast

**Purpose:** Differentiable rasterization for mesh rendering

**AMD Status:** ❌ NOT SUPPORTED natively

**Solution:** Use nvdiffrast-hip from TRELLIS-AMD v1

**Known Limitations (from v1):**
- Multi-bin race conditions at resolutions >128px
- Need AMD-specific coarse rasterizer (CoarseRasterAMD.inl)
- Resolution limiting via `TRELLIS_AMD_RASTER_RES` env var

**Porting Effort:** LOW - Already ported in v1, copy extensions

---

### 2.5 nvdiffrec (External)

**Repository:** https://github.com/NVlabs/nvdiffrec (renderutils branch)

**Purpose:** Split-sum PBR rendering, environment map lighting

**AMD Status:** ❌ NOT SUPPORTED

**Used For:**
- PBR material rendering
- Environment map sampling
- Specular/roughness calculations

**Fallback Option:**
1. Pure OpenGL PBR renderer (would need to write)
2. Simplified material rendering (no roughness/metallic maps)

**Porting Effort:** HIGH - Need to write HIP port or OpenGL fallback

---

## 3. Attention Mechanism Analysis

### 3.1 Backend Configuration

```python
# trellis2/modules/attention/config.py
ATTN_BACKEND = os.environ.get('ATTN_BACKEND', 'flash_attn')
# Options: 'xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive'
```

### 3.2 Compatibility Matrix

| Backend | NVIDIA | AMD ROCm | Notes |
|---------|--------|----------|-------|
| flash_attn | ✅ | ⚠️ | Needs ROCm fork, MI300 only |
| flash_attn_3 | ✅ | ❌ | No AMD support |
| xformers | ✅ | ⚠️ | Partial ROCm support |
| sdpa | ✅ | ✅ | PyTorch native, works |
| naive | ✅ | ✅ | Slow but works |

### 3.3 Critical Issue: Sparse Attention

```python
# trellis2/modules/sparse/config.py
SPARSE_ATTN_BACKEND = os.environ.get('SPARSE_ATTN_BACKEND', None)
# Falls back to ATTN_BACKEND if None
```

**Problem:** Sparse attention only supports `flash_attn` and `xformers`
**No SDPA fallback exists for sparse tensors!**

```python
# From sparse/attention/full_attn.py
if ATTN_BACKEND in ['xformers', 'flash_attn', 'flash_attn_3']:
    # Use memory-efficient attention
else:
    raise NotImplementedError(f"Sparse attention with {ATTN_BACKEND}")
```

**Fix Required:** Implement SDPA fallback for sparse attention

---

## 4. Reusable Fixes from TRELLIS-AMD v1

### 4.1 DINOv2 CPU Fallback (HIGH PRIORITY)

**Problem:** DINOv2 on AMD causes hipBLAS/Tensile kernel crashes

**Solution from v1:**
```python
# Move DINOv2 to CPU during encoding
if hasattr(torch.version, 'hip') and torch.version.hip is not None:
    self.image_cond_model.to('cpu')
    image = image.to('cpu')
    cond = self.image_cond_model(image)
    cond = {k: v.to('cuda') for k, v in cond.items()}
```

**Apply to:** `trellis2/modules/image_feature_extractor.py`

### 4.2 Torchsparse Warmup (MEDIUM PRIORITY)

**Problem:** First torchsparse operation after DINOv2 crashes

**Solution from v1:**
```python
def warmup_torchsparse():
    dummy_sparse = SparseTensor(
        feats=torch.randn(100, 32).cuda(),
        coords=torch.randint(0, 64, (100, 4)).int().cuda()
    )
    _ = sparse_conv(dummy_sparse)
```

### 4.3 Flow Model Precision (MEDIUM PRIORITY)

**Problem:** Mixed precision causes HIP kernel failures

**Solution from v1:**
```python
# Force float32 for flow models on AMD
if self._is_amd:
    self.slat_flow_model.to(torch.float32)
```

### 4.4 Pure OpenGL Mesh Renderer (ALREADY IMPLEMENTED IN v1)

**Files to copy:**
- `trellis/renderers/mesh_renderer_gl.py`
- `trellis/utils/rasterize_gl.py`

---

## 5. New Challenges in TRELLIS.2

### 5.1 Cascade Generation

TRELLIS.2 uses progressive resolution (512→1024→1536):
- Each cascade step requires separate flow model execution
- Memory management is more complex
- More opportunities for AMD-specific failures

**Mitigation:** Apply precision fixes to all cascade stages

### 5.2 PBR Material System

New PBR attributes (base_color, metallic, roughness, alpha):
- Requires texture derivatives for proper rendering
- nvdiffrec uses split-sum approximation
- No existing HIP port

**Mitigation Options:**
1. Port nvdiffrec to HIP (HIGH EFFORT)
2. Implement simplified PBR in OpenGL (MEDIUM EFFORT)
3. Use diffuse-only materials on AMD (LOW EFFORT, quality loss)

### 5.3 Flexible Dual Grid (FDG)

New mesh extraction algorithm:
- More complex than simple marching cubes
- Produces higher quality meshes
- Uses O-Voxel for storage

**Status:** O-Voxel has HIP support, FDG should work

---

## 6. Porting Strategy

### Phase 1: Basic Inference (Week 1-2)

1. **Copy nvdiffrast-hip from v1**
   - Replace `pip install nvdiffrast` with local extension
   - Apply 128px resolution limit

2. **Add torchsparse backend**
   - Set `SPARSE_CONV_BACKEND=torchsparse`
   - Copy torchsparse extension from v1

3. **Implement DINOv2 CPU fallback**
   - Modify `image_feature_extractor.py`
   - Add AMD detection and CPU offloading

4. **Set attention backend**
   - Set `ATTN_BACKEND=sdpa` for dense attention
   - Need to implement sparse SDPA fallback

### Phase 2: Full Quality (Week 3-4)

5. **Implement sparse SDPA**
   - Add SDPA path to `sparse/attention/full_attn.py`
   - Handle variable-length tensors

6. **Port CuMesh or add fallback**
   - Test CuMesh with HIP
   - If fails, use Trimesh CPU fallback

7. **Test cascade generation**
   - Apply precision fixes to all stages
   - Test 1024 and 1536 resolutions

### Phase 3: PBR Support (Week 5-6)

8. **Evaluate nvdiffrec porting**
   - Analyze HIP compatibility
   - If too complex, implement OpenGL PBR

9. **Add pure OpenGL PBR renderer**
   - Environment map loading
   - Split-sum approximation
   - Material texture sampling

---

## 7. Environment Variable Summary

```bash
# Attention
export ATTN_BACKEND=sdpa              # Use PyTorch SDPA
export SPARSE_ATTN_BACKEND=sdpa       # Use SDPA for sparse (needs implementation)

# Sparse convolution
export SPARSE_CONV_BACKEND=torchsparse  # Use torchsparse instead of flex_gemm

# Rendering
export TRELLIS_AMD_RASTER_RES=128     # Limit nvdiffrast resolution
export TRELLIS_MESH_BACKEND=gl        # Use pure OpenGL renderer

# Texture quality
export TRELLIS_TEXTURE_SIZE=1024      # Output texture resolution
export TRELLIS_TEXTURE_MODE=fast      # Fast mode for AMD
```

---

## 8. Risk Assessment

### Critical Blockers

1. **FlexGEMM has no HIP support**
   - Must use torchsparse as fallback
   - May have performance impact

2. **Sparse attention has no SDPA fallback**
   - Must implement or model won't run
   - Affects all flow models

3. **nvdiffrec is CUDA-only**
   - PBR materials won't render correctly
   - Need OpenGL fallback for full feature parity

### Medium Risks

4. **CuMesh HIP compatibility unknown**
   - May need Trimesh fallback (slower)

5. **Cascade generation complexity**
   - Multiple stages = more failure points
   - Memory management more critical

### Low Risks

6. **O-Voxel has HIP support**
   - Should work with minimal changes

7. **Core PyTorch operations**
   - ROCm compatibility is mature

---

## 9. Recommended Starting Point

1. Clone this repo to `TRELLIS.2-AMD`
2. Copy extensions from `TRELLIS-AMD/extensions/`:
   - `nvdiffrast-hip/`
   - `torchsparse/`
   - `diff-gaussian-rasterization/` (if needed for Gaussian output)
3. Create `install_amd.sh` based on v1
4. Set environment variables for AMD backends
5. Implement sparse SDPA fallback
6. Test basic inference at 512³

---

## 10. Files to Modify

### High Priority
- [ ] `trellis2/modules/image_feature_extractor.py` - DINOv2 CPU fallback
- [ ] `trellis2/modules/sparse/attention/full_attn.py` - SDPA fallback
- [ ] `trellis2/modules/sparse/config.py` - Add torchsparse detection
- [ ] `setup.sh` → `install_amd.sh` - AMD installation script

### Medium Priority
- [ ] `trellis2/pipelines/trellis2_image_to_3d.py` - AMD warmup/precision
- [ ] `trellis2/renderers/mesh_renderer.py` - Resolution limiting
- [ ] `trellis2/renderers/pbr_mesh_renderer.py` - OpenGL fallback

### Low Priority
- [ ] `o-voxel/o_voxel/postprocess.py` - GLB export fixes
- [ ] `app.py` - Environment variable documentation

---

## Appendix A: Extension Source Code Locations

```
TRELLIS.2/
├── o-voxel/
│   └── src/
│       ├── hash/hash.cu          # GPU hashmap
│       ├── rasterize/rasterize.cu # Voxel rendering
│       └── serialize/*.cu        # Spatial encoding

TRELLIS-AMD/extensions/ (copy from v1)
├── nvdiffrast-hip/
│   └── csrc/common/
│       ├── hipraster/impl/       # HIP rasterizer
│       ├── *.hip                 # HIP kernel files
│       └── *.cu                  # CUDA kernels (reference)
├── torchsparse/
│   └── torchsparse/backend/
│       ├── *.hip                 # HIP sparse ops
│       └── *.cu                  # CUDA sparse ops
└── diff-gaussian-rasterization/
    ├── cuda_rasterizer/          # CUDA kernels
    └── hip_rasterizer/           # HIP kernels
```

---

*Document generated: 2025-12-31*
*Based on analysis of TRELLIS.2 main branch and TRELLIS-AMD amd-wsl2-fixes branch*
