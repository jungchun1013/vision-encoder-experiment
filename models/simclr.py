import torch.nn as nn

from models.registry import register
from wrappers.encoder import BaseEncoder


@register("simclr")
class SimCLREncoder(BaseEncoder):
    name = "SimCLR"
    feature_dim = 2048

    def load_model(self) -> nn.Module:
        raise NotImplementedError(
            "SimCLR requires a manual checkpoint download.\n"
            "1. Download the ResNet-50 (1x) checkpoint from:\n"
            "   https://github.com/google-research/simclr\n"
            "2. Convert to PyTorch format and place in checkpoints/simclr_r50_1x.pth"
        )

    def get_transform(self):
        raise NotImplementedError("SimCLR encoder not yet implemented.")
