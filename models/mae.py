import torch
import torch.nn as nn
from transformers import ViTMAEModel, ViTMAEConfig
from torchvision import transforms

from models.registry import register
from wrappers.encoder import BaseEncoder

MODEL_ID = "facebook/vit-mae-base"


@register("mae")
class MAEEncoder(BaseEncoder):
    name = "MAE"
    feature_dim = 768

    def load_model(self) -> nn.Module:
        config = ViTMAEConfig.from_pretrained(MODEL_ID)
        config.mask_ratio = 0.0
        return ViTMAEModel.from_pretrained(MODEL_ID, config=config)

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images)
        return output.last_hidden_state[:, 0]  # CLS token
