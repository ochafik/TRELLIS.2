# AMD HIP Extension Build Status

This document tracks the status of building TRELLIS.2 CUDA extensions for AMD GPUs using ROCm/HIP on Windows.

## Build Environment

- **OS**: Windows 11
- **ROCm Version**: 6.4
- **PyTorch**: 2.11.0a0+rocm7.11.0a20251218
- **GPU Architecture**: gfx1151 (AMD Radeon 8060S Strix Point APU)
- **Python**: 3.12

## Extension Build Status

| Extension | Status | Notes |
|-----------|--------|-------|
| flex_gemm | ✅ Built | Loads successfully; triton kernels require triton package |
| o-voxel | ✅ Built | All fixes applied, loads successfully |
| cumesh | ✅ **FIXED** | All operations working with CPU copy workaround |
| torchsparse | ✅ Works | Pre-built, AMD HIP compatible |
| diff-gaussian-rasterization | ⚠️ Untested | Likely needs similar fixes |
| nvdiffrast-hip | ✅ Working | HIP rasterization works for mesh rendering |
| xatlas | ✅ Working | UV unwrapping works (part of cumesh) |

## Fixes Applied

### o-voxel Extension

1. **setup.py** - Added HIP include paths and compiler fixes:
   - Added `HIP_INCLUDE` path with Windows short path conversion
   - Added `PYTHON_INCLUDE` with short path conversion
   - Added `-DSTRIP_ERROR_MESSAGES` for c10::ValueError ABI compatibility
   - Changed `ext.cpp` to `ext.cu` so hipcc compiles it instead of MSVC

2. **src/io/filter_parent.cpp** - Fixed narrowing conversions:
   ```cpp
   // Before
   torch::Tensor delta = torch::zeros({N_leaf, C}, torch::kUInt8);
   // After
   torch::Tensor delta = torch::zeros({static_cast<int64_t>(N_leaf), static_cast<int64_t>(C)}, torch::kUInt8);
   ```

3. **src/io/filter_neighbor.cpp** - Same narrowing conversion fixes

4. **src/io/svo.cpp** - Fixed `torch::from_blob` narrowing conversions:
   ```cpp
   // Before
   torch::Tensor svo_tensor = torch::from_blob(svo.data(), {svo.size()}, torch::kUInt8).clone();
   // After
   torch::Tensor svo_tensor = torch::from_blob(svo.data(), {static_cast<int64_t>(svo.size())}, torch::kUInt8).clone();
   ```

5. **src/convert/flexible_dual_grid.cpp** - Fixed non-standard literal suffixes:
   ```cpp
   // Before (non-standard 'd' suffix)
   if (segment_length < 1e-6d) continue;
   if (dir[axis] == 0.0d) {
   // After (standard C++)
   if (segment_length < 1e-6) continue;
   if (dir[axis] == 0.0) {
   ```

6. **src/ext.cu** - Created from ext.cpp to ensure hipcc compilation

### flex_gemm Extension

1. **setup.py** - Added same HIP include path and compiler fixes as o-voxel
2. Extension builds and loads successfully
3. Requires triton package for full kernel functionality

## cumesh Fix (RESOLVED)

The cumesh extension had multiple issues on AMD HIP/gfx1151 that have been resolved:

### Issue 1: hipMemcpy D2D Kernel Trigger Crash (FIXED)

**Root Cause**: `hipMemcpy` with `hipMemcpyDeviceToDevice` internally uses a GPU kernel on HIP. This triggered lazy loading of **all** cumesh kernels, including some hipcub-based kernels that fail to compile for gfx1151 (RDNA3 Strix Point).

**Fix Applied** in `extensions/cumesh/src/io.cu`:

1. **`CuMesh::init()`** - Changed from D2D copy to CPU roundtrip:
   ```cpp
   // WORKAROUND for AMD HIP/gfx1151: hipMemcpy D2D internally uses a kernel,
   // which triggers lazy loading of all cumesh kernels. Some hipcub-based
   // kernels fail to compile for gfx1151, causing a crash. To avoid this,
   // we copy via CPU (H2D) instead of D2D.
   if (num_vertices > 0) {
       auto v_cpu = vertices.contiguous().cpu();
       CUDA_CHECK(cudaMemcpy(this->vertices.ptr, v_cpu.data_ptr<float>(),
                            num_vertices * sizeof(float3), cudaMemcpyHostToDevice));
   }
   ```

2. **`buffer_to_tensor()`** - Changed from D2D copy to CPU roundtrip:
   ```cpp
   // Create CPU tensor first, copy D2H, then move to GPU using PyTorch
   auto cpu_tensor = torch::empty(shape, cpu_options);
   CUDA_CHECK(cudaMemcpy(cpu_tensor.data_ptr(), buffer.ptr,
                        count * sizeof(T), cudaMemcpyDeviceToHost));
   return cpu_tensor.to(torch::kCUDA);  // PyTorch handles the H2D copy
   ```

### Issue 2: CUB→hipcub API Compatibility (Build-time fixes)

Several build-time fixes were applied for hipcub/rocprim compatibility:

1. **int3_decomposer**: Changed from `thrust::tuple` to `rocprim::tuple`
2. **DeviceSegmentedReduce**: Added Float3Add operator for float3 types
3. **ROCPRIM_WAVEFRONT_SIZE**: Set to 32 for RDNA3 wave32 architecture

### Verified Working Operations

All cumesh operations now work on AMD HIP:
- `init()`, `read()` - Basic mesh I/O
- `compute_face_normals()`, `compute_vertex_normals()` - Normal computation
- `get_edges()`, `get_boundary_info()`, `get_connected_components()` - Mesh analysis
- `fill_holes(max_hole_perimeter)` - Hole filling
- `simplify(target_num_faces)` - Mesh decimation
- `uv_unwrap()` via xatlas - UV parameterization

The performance impact is minimal since these operations are typically done once per mesh, and the CPU roundtrip adds only a few milliseconds.

## Runtime Requirements

To load the built extensions, the following DLL directories must be in the search path:

```python
import os

# Add ROCm bin directory
os.add_dll_directory(r"C:\Program Files\AMD\ROCm\6.4\bin")

# Add PyTorch lib directory
os.add_dll_directory(r"C:\Dev\TRELLIS.2\.venv\Lib\site-packages\torch\lib")

# Import torch first to initialize its DLLs
import torch

# Now extensions can be imported
import flex_gemm
import o_voxel
```

## TRELLIS.2 Framework Status

The TRELLIS.2 framework correctly detects AMD HIP and configures appropriate backends:

```
[SPARSE] Platform: AMD HIP; Conv backend: torchsparse; Attention backend: sdpa
[AMD] DINOv2/v3 will use CPU (set TRELLIS_DINO_CPU=0 to force GPU)
[ATTENTION] Using backend: flash_attn
[Pipeline] Platform: AMD HIP
[Pipeline] GPU: AMD Radeon(TM) 8060S Graphics
[Pipeline] AMD-specific settings:
  - ATTN_BACKEND: auto (sdpa)
  - SPARSE_CONV_BACKEND: auto (torchsparse)
  - TRELLIS_DINO_CPU: 1 (CPU mode)
```

## Build Scripts

Several build scripts were created for building individual extensions:

- `build_flex.ps1` - Build flex_gemm extension
- `build_ovoxel.ps1` - Build o-voxel extension
- `build_cumesh.ps1` - Build cumesh extension (currently blocked)
- `build_all_ext.ps1` - Build all extensions
- `install_ext.ps1` - Install all extensions

## Next Steps

1. ~~**cumesh**: Requires significant refactoring for hipcub compatibility~~ ✅ **DONE**
2. **triton**: Investigate AMD ROCm triton support or fallbacks
3. **E2E Testing**: Full inference testing with image→GLB pipeline
4. ~~**nvdiffrast-hip**: Test and verify HIP rasterization works~~ ✅ **DONE**
5. **Visual Quality**: Verify exported GLB files render correctly
