#!/usr/bin/env python
"""Test TRELLIS.2 inference on AMD GPU."""
import os
import sys

# Add ROCm bin to DLL search path
rocm_bin = r"C:\Program Files\AMD\ROCm\6.4\bin"
if os.path.exists(rocm_bin):
    os.add_dll_directory(rocm_bin)

# Add PyTorch lib to DLL search path
torch_lib = r"C:\Dev\TRELLIS.2\.venv\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

# Add extension directories to Python path
sys.path.insert(0, r"C:\Dev\TRELLIS.2")
sys.path.insert(0, r"C:\Dev\TRELLIS.2\extensions\flex_gemm")
sys.path.insert(0, r"C:\Dev\TRELLIS.2\o-voxel")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\n=== Loading TRELLIS.2 Pipeline ===")
from trellis2.pipelines import Trellis2ImageTo3DPipeline

print("\n=== Loading from pretrained ===")
try:
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    print(f"Pipeline loaded: {pipeline}")
    print("SUCCESS!")
except Exception as e:
    import traceback
    print(f"FAILED: {e}")
    traceback.print_exc()
