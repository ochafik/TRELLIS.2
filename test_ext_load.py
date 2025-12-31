import os
import sys

# Add ROCm bin to DLL search path
rocm_bin = r"C:\Program Files\AMD\ROCm\6.4\bin"
if os.path.exists(rocm_bin):
    os.add_dll_directory(rocm_bin)
    print(f"Added DLL directory: {rocm_bin}")

# Add PyTorch lib to DLL search path
torch_lib = r"C:\Dev\TRELLIS.2\.venv\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    print(f"Added DLL directory: {torch_lib}")

# Import torch first to initialize its DLLs
import torch
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

# Test flex_gemm
print("\nTesting flex_gemm:")
sys.path.insert(0, r"C:\Dev\TRELLIS.2\extensions\flex_gemm\flex_gemm\kernels")
try:
    import cuda as flex_gemm_cuda
    print(f"  SUCCESS: {flex_gemm_cuda}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test o_voxel
print("\nTesting o_voxel:")
sys.path.insert(0, r"C:\Dev\TRELLIS.2\o-voxel\o_voxel")
try:
    import _C as o_voxel_C
    print(f"  SUCCESS: {o_voxel_C}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nDone!")
