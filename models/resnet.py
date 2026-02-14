import timm
import timm.data
import torch.nn as nn

from models.registry import register
from wrappers.encoder import BaseEncoder

MODEL_ID = "resnet50"


@register("resnet")
class ResNetEncoder(BaseEncoder):
    name = "ResNet-50"
    feature_dim = 2048

    def load_model(self) -> nn.Module:
        return timm.create_model(MODEL_ID, pretrained=True, num_classes=0)

    def get_transform(self):
        data_cfg = timm.data.resolve_model_data_config(self.model)
        return timm.data.create_transform(**data_cfg, is_training=False)
