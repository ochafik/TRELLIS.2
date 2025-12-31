from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os
import sysconfig
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

# Get HIP include path for Windows ROCm builds
# Use short path format to avoid issues with spaces in paths
HIP_INCLUDE = None
if platform.system() == "Windows":
    rocm_path = os.environ.get("ROCM_PATH", os.environ.get("HIP_PATH", r"C:\Program Files\AMD\ROCm\6.4"))
    if os.path.exists(rocm_path):
        HIP_INCLUDE = os.path.join(rocm_path, "include")
        if ' ' in HIP_INCLUDE:
            import ctypes
            buf = ctypes.create_unicode_buffer(512)
            if ctypes.windll.kernel32.GetShortPathNameW(HIP_INCLUDE, buf, len(buf)):
                HIP_INCLUDE = buf.value

# Get Python include path (needed for HIP builds on Windows)
# Use short path format to avoid issues with spaces in paths on Windows
PYTHON_INCLUDE = sysconfig.get_path('include')
if platform.system() == "Windows" and ' ' in PYTHON_INCLUDE:
    # Convert to short path format (8.3)
    import ctypes
    buf = ctypes.create_unicode_buffer(512)
    if ctypes.windll.kernel32.GetShortPathNameW(PYTHON_INCLUDE, buf, len(buf)):
        PYTHON_INCLUDE = buf.value

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True

if not IS_HIP:
    cc_flag = []
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

# Platform-specific compile args
# Windows ROCm needs STRIP_ERROR_MESSAGES to avoid c10::ValueError linker issues
if platform.system() == "Windows":
    # /wd2398 disables narrowing conversion errors in initializer lists
    extra_compile_args_cxx = ["/O2", "/std:c++17", "/EHsc", "-DSTRIP_ERROR_MESSAGES"]
    extra_compile_args_nvcc = ["-O3", "-std=c++17", "-DSTRIP_ERROR_MESSAGES"] + cc_flag
else:
    extra_compile_args_cxx = ["-O3", "-std=c++17"]
    extra_compile_args_nvcc = ["-O3", "-std=c++17"] + cc_flag

setup(
    name="o_voxel",
    packages=[
        'o_voxel',
        'o_voxel.convert',
        'o_voxel.io',
    ],
    ext_modules=[
        CUDAExtension(
            name="o_voxel._C",
            sources=[
                # Hashmap functions
                "src/hash/hash.cu",
                # Convert functions
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetic_attr.cpp",
                ## Serialization functions
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                # IO functions
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                # Rasterization functions
                "src/rasterize/rasterize.cu",
                
                # main (use .cu so hipcc compiles it, not MSVC)
                "src/ext.cu",
            ],
            include_dirs=[d for d in [
                os.path.join(ROOT, "third_party/eigen"),
                PYTHON_INCLUDE,
                HIP_INCLUDE,
            ] if d is not None],
            extra_compile_args={
                "cxx": extra_compile_args_cxx,
                "nvcc": extra_compile_args_nvcc,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
