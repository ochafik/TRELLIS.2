from typing import *
import os
import torch
import torch.nn as nn
from .. import models

# AMD HIP detection
_IS_AMD = hasattr(torch.version, 'hip') and torch.version.hip is not None


class Pipeline:
    """
    A base class for pipelines.

    AMD HIP Note: On AMD GPUs, the pipeline will automatically use AMD-compatible
    backends (SDPA for attention, torchsparse for sparse convolutions). A warmup
    step is recommended before first inference to initialize the sparse kernels.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()
        self._is_amd = _IS_AMD
        self._warmup_done = False

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                _models[k] = models.from_pretrained(v)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args

        # Print platform info on first load
        if _IS_AMD:
            cls.print_platform_info()

        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))

    def warmup(self) -> None:
        """
        Perform a warmup to initialize sparse kernels.

        AMD HIP Note: On AMD GPUs, torchsparse kernels need to be compiled/cached
        on first use. Running warmup() before inference avoids startup delays.
        """
        if self._warmup_done:
            return

        if self._is_amd:
            print("[AMD] Running warmup for torchsparse kernels...")

        try:
            from ..modules.sparse import SparseTensor
            # Create a small sparse tensor to trigger kernel compilation
            dummy_coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]], device=self.device, dtype=torch.int)
            dummy_feats = torch.randn(3, 64, device=self.device)
            dummy_sparse = SparseTensor(feats=dummy_feats, coords=dummy_coords)
            # Trigger sparse operations
            _ = dummy_sparse.feats.sum()
            del dummy_sparse, dummy_coords, dummy_feats
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            if self._is_amd:
                print(f"[AMD] Warmup warning: {e}")

        self._warmup_done = True
        if self._is_amd:
            print("[AMD] Warmup complete")

    @staticmethod
    def print_platform_info() -> None:
        """Print platform and backend information."""
        platform = "AMD HIP" if _IS_AMD else "NVIDIA CUDA"
        print(f"[Pipeline] Platform: {platform}")
        if torch.cuda.is_available():
            print(f"[Pipeline] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Pipeline] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        if _IS_AMD:
            print("[Pipeline] AMD-specific settings:")
            print(f"  - ATTN_BACKEND: {os.environ.get('ATTN_BACKEND', 'auto (sdpa)')}")
            print(f"  - SPARSE_CONV_BACKEND: {os.environ.get('SPARSE_CONV_BACKEND', 'auto (torchsparse)')}")
            print(f"  - TRELLIS_DINO_CPU: {os.environ.get('TRELLIS_DINO_CPU', '1 (CPU mode)')}")