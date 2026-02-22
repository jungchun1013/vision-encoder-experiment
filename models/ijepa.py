import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from torchvision import transforms

from models.registry import register
from wrappers.encoder import BaseEncoder

MODEL_ID = "facebook/ijepa_vith14_1k"


@register("ijepa")
class IJEPAEncoder(BaseEncoder):
    name = "I-JEPA"
    feature_dim = 1280  # ViT-H hidden_size

    def load_model(self) -> nn.Module:
        return AutoModel.from_pretrained(MODEL_ID)

    def get_transform(self):
        # Use the HuggingFace processor's image_mean/std but return a
        # standard torchvision transform (consistent with other encoders)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        mean = processor.image_mean
        std = processor.image_std
        size = processor.size.get("height", 224)
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images)
        # I-JEPA has no CLS token — average pool all patch tokens
        return output.last_hidden_state.mean(dim=1)  # [B, 1280]
