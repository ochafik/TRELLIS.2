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
| cumesh | ❌ Blocked | CUB→hipcub API incompatibility with custom types |
| torchsparse | ✅ Works | Pre-built, AMD HIP compatible |
| diff-gaussian-rasterization | ⚠️ Untested | Likely needs similar fixes |
| nvdiffrast-hip | ⚠️ Untested | HIP-specific version exists |

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

## cumesh Blockers

The cumesh extension uses CUB device algorithms with custom types that are incompatible with hipcub/rocprim:

### Issue 1: DeviceRadixSort with Custom Decomposer

```cpp
struct int3_decomposer {
    __host__ __device__ thrust::tuple<int&, int&, int&> operator()(int3& key) const {
        return thrust::make_tuple(std::ref(key.x), std::ref(key.y), std::ref(key.z));
    }
};
```

**Error**: rocprim doesn't understand `thrust::tuple` - it expects `rocprim::tuple`:
```
error: no matching function for call to 'tuple_bit_size_impl'
note: implicit instantiation of undefined template 'rocprim::tuple_size<thrust::tuple<int &, int &, int &>>'
```

### Issue 2: DeviceSegmentedReduce with Custom Vec3f

```cpp
struct Vec3f {
    float x, y, z;
    // ... operators
};
```

**Error**: rocprim's segmented reduce doesn't support custom types with non-trivial constructors:
```
error: no matching constructor for initialization of 'input_type' (aka 'cumesh::Vec3f')
```

### Potential Solutions

1. **For int3_decomposer**: Use `rocprim::tuple` instead of `thrust::tuple`
2. **For Vec3f**: Either:
   - Use `float3` (POD type) instead of custom Vec3f
   - Implement custom reduction kernels
   - Do component-wise scalar reductions

These changes would require significant refactoring across multiple files.

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

1. **cumesh**: Requires significant refactoring for hipcub compatibility
2. **triton**: Investigate AMD ROCm triton support or fallbacks
3. **Testing**: Full inference testing once all extensions are available
4. **nvdiffrast-hip**: Test and verify HIP rasterization works
