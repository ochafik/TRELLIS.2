from typing import *
import torch

CONV = 'flex_gemm'
DEBUG = False
ATTN = 'flash_attn'

# AMD GPU detection
_IS_AMD = hasattr(torch.version, 'hip') and torch.version.hip is not None

def __from_env():
    import os

    global CONV
    global DEBUG
    global ATTN

    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    # AMD HIP: Auto-detect best backends for AMD GPUs
    if _IS_AMD:
        # Default to torchsparse on AMD (flex_gemm is CUDA-only)
        if env_sparse_conv_backend is None:
            env_sparse_conv_backend = 'torchsparse'
        # Default to sdpa on AMD (flash_attn needs ROCm fork)
        if env_sparse_attn_backend is None:
            env_sparse_attn_backend = 'sdpa'

    # Valid backends now include 'sdpa' and 'naive' for AMD compatibility
    valid_conv_backends = ['none', 'spconv', 'torchsparse', 'flex_gemm']
    valid_attn_backends = ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in valid_conv_backends:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in valid_attn_backends:
        ATTN = env_sparse_attn_backend

    platform = "AMD HIP" if _IS_AMD else "NVIDIA CUDA"
    print(f"[SPARSE] Platform: {platform}; Conv backend: {CONV}; Attention backend: {ATTN}")
        

__from_env()
    

def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']):
    global ATTN
    ATTN = backend
