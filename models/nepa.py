import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms

from models.registry import register
from wrappers.encoder import BaseEncoder

MODEL_ID = "SixAILab/nepa-base-patch14-224"


@register("nepa")
class NEPAEncoder(BaseEncoder):
    name = "NEPA"
    feature_dim = 768  # ViT-B hidden_size

    def load_model(self) -> nn.Module:
        return AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

    def get_transform(self):
        # mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], size=224
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images)
        # NEPA is causal (autoregressive) — mean pool all patch tokens
        return output.last_hidden_state.mean(dim=1)  # [B, 768]
