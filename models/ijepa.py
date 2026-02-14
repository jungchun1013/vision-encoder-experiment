import torch.nn as nn

from models.registry import register
from wrappers.encoder import BaseEncoder


@register("ijepa")
class IJEPAEncoder(BaseEncoder):
    name = "I-JEPA"
    feature_dim = 768

    def load_model(self) -> nn.Module:
        raise NotImplementedError(
            "I-JEPA requires cloning the repo and downloading a checkpoint.\n"
            "1. git clone https://github.com/facebookresearch/ijepa\n"
            "2. Download ViT-B/16 checkpoint from the repo's model zoo\n"
            "3. Use the target_encoder weights, average-pool patch tokens (no CLS token)"
        )

    def get_transform(self):
        raise NotImplementedError("I-JEPA encoder not yet implemented.")
