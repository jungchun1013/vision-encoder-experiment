import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from torchvision import transforms

from models.registry import register
from wrappers.encoder import BaseEncoder

MODEL_ID = "openai/clip-vit-large-patch14-336"


@register("llava")
class LLaVAEncoder(BaseEncoder):
    name = "LLaVA"
    feature_dim = 1024  # ViT-L hidden_size

    def load_model(self) -> nn.Module:
        return CLIPVisionModel.from_pretrained(MODEL_ID)

    def get_transform(self):
        processor = CLIPImageProcessor.from_pretrained(MODEL_ID)
        size = processor.crop_size["height"]
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images, output_hidden_states=True)
        # LLaVA uses second-to-last hidden state, CLS token removed
        hidden = output.hidden_states[-2]  # [B, 577, 1024]
        patches = hidden[:, 1:]            # [B, 576, 1024]
        return patches.mean(dim=1)         # [B, 1024]
