from typing import *
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image

# AMD HIP detection
_IS_AMD = hasattr(torch.version, 'hip') and torch.version.hip is not None
# Allow override via environment variable
_DINO_USE_CPU = os.environ.get('TRELLIS_DINO_CPU', '1' if _IS_AMD else '0') == '1'

if _IS_AMD:
    print(f"[AMD] DINOv2/v3 will use {'CPU' if _DINO_USE_CPU else 'GPU'} (set TRELLIS_DINO_CPU=0 to force GPU)")


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.

    AMD HIP Note: DINOv2 has issues with hipBLAS/Tensile on AMD GPUs.
    By default on AMD, the model runs on CPU and results are moved back to GPU.
    Set TRELLIS_DINO_CPU=0 to force GPU execution (may crash on some AMD GPUs).
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._is_amd = _IS_AMD
        self._use_cpu = _DINO_USE_CPU

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.

        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.

        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        # Determine target device
        target_device = 'cuda'

        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            target_device = image.device
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image)
            if not self._use_cpu:
                image = image.cuda()
                target_device = 'cuda'
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        # AMD HIP: Run DINOv2 on CPU to avoid hipBLAS issues
        if self._use_cpu:
            # Move model and image to CPU
            self.model.cpu()
            image_cpu = self.transform(image).cpu()
            features = self.model(image_cpu, is_training=True)['x_prenorm']
            patchtokens = F.layer_norm(features, features.shape[-1:])
            # Move results back to GPU
            return patchtokens.to(target_device)
        else:
            # Standard GPU path
            image = self.transform(image).cuda()
            features = self.model(image, is_training=True)['x_prenorm']
            patchtokens = F.layer_norm(features, features.shape[-1:])
            return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.

    AMD HIP Note: DINOv3 may have similar issues to DINOv2 on AMD GPUs.
    By default on AMD, the model runs on CPU and results are moved back to GPU.
    Set TRELLIS_DINO_CPU=0 to force GPU execution (may crash on some AMD GPUs).
    """
    def __init__(self, model_name: str, image_size=512):
        self.model_name = model_name
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._is_amd = _IS_AMD
        self._use_cpu = _DINO_USE_CPU

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.

        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.

        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        # Determine target device
        target_device = 'cuda'

        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            target_device = image.device
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image)
            if not self._use_cpu:
                image = image.cuda()
                target_device = 'cuda'
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        # AMD HIP: Run DINOv3 on CPU to avoid hipBLAS issues
        if self._use_cpu:
            # Move model and image to CPU
            self.model.cpu()
            image_cpu = self.transform(image).cpu()
            features = self.extract_features(image_cpu)
            # Move results back to GPU
            return features.to(target_device)
        else:
            # Standard GPU path
            image = self.transform(image).cuda()
            features = self.extract_features(image)
            return features
