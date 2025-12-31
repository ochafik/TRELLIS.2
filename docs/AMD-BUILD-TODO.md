# TRELLIS.2 AMD Port - Build TODO

## Status Overview

| Component | Status | Notes |
|-----------|--------|-------|
| Core Python changes | Done | All 7 tests pass |
| install_amd.ps1 | Done | Fixed error handling |
| run_amd.ps1 | Done | Sets all AMD env vars |
| o-voxel extension | In Progress | HIP build failing |
| cumesh extension | In Progress | HIP build failing |
| flex_gemm extension | In Progress | HIP build failing |
| nvdiffrast-hip | Copied | From TRELLIS-AMD v1 |
| torchsparse | Copied | From TRELLIS-AMD v1 |

## TODO

### High Priority
- [ ] Debug o-voxel HIP compilation errors
- [ ] Debug cumesh HIP compilation errors
- [ ] Debug flex_gemm HIP compilation errors
- [ ] Test full inference pipeline

### Medium Priority
- [ ] Optimize DINOv2 (try aotriton instead of CPU fallback)
- [ ] Performance benchmarking vs NVIDIA
- [ ] Memory usage optimization

### Low Priority
- [ ] Port nvdiffrec for full PBR pipeline
- [ ] Investigate native HIP cumesh port

## Extension Build Status

### o-voxel
- **Hipify**: Working (13 kernel launches replaced)
- **Compile**: FAILING
- **Error**: TBD - need to capture full build log

### cumesh
- **Hipify**: Working (102 kernel launches replaced)
- **Compile**: FAILING
- **Error**: TBD - need to capture full build log

### flex_gemm
- **Hipify**: Working (15 kernel launches replaced)
- **Compile**: FAILING
- **Error**: TBD - need to capture full build log

## Test Results (2024-12-31)

```
============================================================
TEST SUMMARY
============================================================
  PyTorch/HIP: PASS
  Sparse Config: PASS
  Sparse Tensor: PASS
  SDPA Attention: PASS
  DINOv2 CPU Fallback: PASS
  OpenGL Renderer: PASS
  Pipeline: PASS

Total: 7 passed, 0 failed
```

## Environment

- OS: Windows 11
- GPU: AMD Radeon 8060S (Strix Point APU)
- ROCm: 7.11.0a20251218 (via PyTorch wheels)
- PyTorch: 2.11.0a0+rocm7.11.0a20251218
- Python: 3.12
