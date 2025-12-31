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

# Add extension directories to Python path
sys.path.insert(0, r"C:\Dev\TRELLIS.2")
sys.path.insert(0, r"C:\Dev\TRELLIS.2\extensions\flex_gemm")
sys.path.insert(0, r"C:\Dev\TRELLIS.2\o-voxel")

print("\n=== Testing PyTorch ===")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

print("\n=== Testing flex_gemm ===")
try:
    import flex_gemm
    print(f"flex_gemm: {flex_gemm}")
except Exception as e:
    print(f"flex_gemm FAILED: {e}")

print("\n=== Testing o_voxel ===")
try:
    import o_voxel
    print(f"o_voxel: {o_voxel}")
except Exception as e:
    print(f"o_voxel FAILED: {e}")

print("\n=== Testing trellis2 import ===")
try:
    import trellis2
    print(f"trellis2: {trellis2}")
except Exception as e:
    print(f"trellis2 FAILED: {e}")

print("\n=== Testing Trellis2ImageTo3DPipeline ===")
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print(f"Trellis2ImageTo3DPipeline: {Trellis2ImageTo3DPipeline}")
except Exception as e:
    import traceback
    print(f"Trellis2ImageTo3DPipeline FAILED: {e}")
    traceback.print_exc()

print("\nDone!")
