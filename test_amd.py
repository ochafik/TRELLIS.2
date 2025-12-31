#!/usr/bin/env python3
"""
TRELLIS.2 AMD Test Script

Tests basic functionality on AMD GPUs with ROCm/HIP.
Run this after installation to verify AMD compatibility.

Usage:
    source .venv/bin/activate  # or activate your venv
    python test_amd.py
"""

import os
import sys

# Set AMD environment variables BEFORE importing torch
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
os.environ.setdefault('SPARSE_ATTN_BACKEND', 'sdpa')
os.environ.setdefault('SPARSE_CONV_BACKEND', 'torchsparse')
os.environ.setdefault('TRELLIS_DINO_CPU', '1')
os.environ.setdefault('XFORMERS_DISABLED', '1')


def test_pytorch():
    """Test PyTorch and HIP detection."""
    print("\n" + "=" * 60)
    print("Test 1: PyTorch and HIP Detection")
    print("=" * 60)

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"HIP version: {getattr(torch.version, 'hip', 'N/A')}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None
    print(f"AMD HIP detected: {is_amd}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Basic tensor test
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        z = torch.matmul(x, y)
        print(f"Basic tensor ops: PASS (result shape: {z.shape})")
    else:
        print("WARNING: No GPU detected!")
        return False

    return True


def test_sparse_config():
    """Test sparse module configuration."""
    print("\n" + "=" * 60)
    print("Test 2: Sparse Module Configuration")
    print("=" * 60)

    try:
        from trellis2.modules.sparse import config
        print(f"Sparse conv backend: {config.CONV}")
        print(f"Sparse attn backend: {config.ATTN}")
        print(f"AMD detected in config: {config._IS_AMD}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_sparse_tensor():
    """Test sparse tensor creation."""
    print("\n" + "=" * 60)
    print("Test 3: Sparse Tensor")
    print("=" * 60)

    try:
        import torch
        from trellis2.modules.sparse import SparseTensor

        # Create a small sparse tensor
        coords = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 2, 2, 2],
        ], device='cuda', dtype=torch.int)
        feats = torch.randn(3, 64, device='cuda')

        sparse = SparseTensor(feats=feats, coords=coords)
        print(f"Sparse tensor created: {sparse.shape}")
        print(f"Sparse tensor device: {sparse.device}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sdpa_attention():
    """Test SDPA attention backend."""
    print("\n" + "=" * 60)
    print("Test 4: SDPA Attention Backend")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F

        # Basic SDPA test
        B, H, L, D = 2, 8, 64, 64
        q = torch.randn(B, H, L, D, device='cuda')
        k = torch.randn(B, H, L, D, device='cuda')
        v = torch.randn(B, H, L, D, device='cuda')

        out = F.scaled_dot_product_attention(q, k, v)
        print(f"SDPA output shape: {out.shape}")
        print("SDPA attention: PASS")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_dino_cpu_fallback():
    """Test DINOv2 CPU fallback detection."""
    print("\n" + "=" * 60)
    print("Test 5: DINOv2 CPU Fallback")
    print("=" * 60)

    try:
        from trellis2.modules import image_feature_extractor as ife
        print(f"AMD detected: {ife._IS_AMD}")
        print(f"DINO CPU mode: {ife._DINO_USE_CPU}")

        if ife._DINO_USE_CPU:
            print("DINOv2 will run on CPU (hipBLAS workaround active)")
        else:
            print("DINOv2 will run on GPU")

        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_mesh_renderer_gl():
    """Test OpenGL mesh renderer availability."""
    print("\n" + "=" * 60)
    print("Test 6: OpenGL Mesh Renderer")
    print("=" * 60)

    try:
        from trellis2.renderers import create_mesh_renderer, MeshRendererGL
        print("MeshRendererGL import: PASS")

        # Test factory function detection
        import torch
        is_amd = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_amd:
            print("Factory will default to OpenGL backend on AMD")
        else:
            print("Factory will default to HIP/CUDA backend")

        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_pipeline_load():
    """Test pipeline loading (optional, requires model download)."""
    print("\n" + "=" * 60)
    print("Test 7: Pipeline Loading (Optional)")
    print("=" * 60)

    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.pipelines.base import Pipeline

        # Just test that the class can be imported and has AMD features
        print("Pipeline class import: PASS")
        print(f"Pipeline has warmup method: {hasattr(Pipeline, 'warmup')}")
        print(f"Pipeline has print_platform_info: {hasattr(Pipeline, 'print_platform_info')}")

        # Call print_platform_info
        Pipeline.print_platform_info()

        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("TRELLIS.2 AMD Compatibility Test Suite")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("PyTorch/HIP", test_pytorch()))
    results.append(("Sparse Config", test_sparse_config()))
    results.append(("Sparse Tensor", test_sparse_tensor()))
    results.append(("SDPA Attention", test_sdpa_attention()))
    results.append(("DINOv2 CPU Fallback", test_dino_cpu_fallback()))
    results.append(("OpenGL Renderer", test_mesh_renderer_gl()))
    results.append(("Pipeline", test_pipeline_load()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\nAll tests passed! TRELLIS.2 is ready for AMD.")
        print("\nTo run inference:")
        print("  python -c \"")
        print("  from trellis2.pipelines import Trellis2ImageTo3DPipeline")
        print("  pipe = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')")
        print("  pipe.cuda()")
        print("  pipe.warmup()  # Recommended on AMD")
        print("  \"")
    else:
        print("\nSome tests failed. Check the output above for details.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
