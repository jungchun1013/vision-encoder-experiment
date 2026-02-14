from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(ABC):
    """Unified interface for all vision encoders."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable encoder name."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of output feature vectors."""

    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load and return the pretrained model."""

    @abstractmethod
    def get_transform(self):
        """Return the torchvision transform pipeline for this encoder."""

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self.load_model().to(self.device).eval()
        return self._model

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images. Returns [B, feature_dim]."""
        images = images.to(self.device)
        return self._forward(images)

    @torch.no_grad()
    def extract_features_from_layer(self, images: torch.Tensor, layer: str) -> torch.Tensor:
        """Extract activations from a specific named layer using a forward hook.

        Args:
            images: input batch [B, C, H, W]
            layer: dot-separated layer name (e.g. 'blocks.5', 'layer3')
                   Use list_layers() to discover available names.

        Returns:
            Activation tensor. For 3D outputs [B, T, D] the spatial/token dims
            are average-pooled to return [B, D].
        """
        images = images.to(self.device)
        module = self._get_submodule(layer)
        activation = {}

        def hook_fn(mod, inp, out):
            activation["out"] = out

        handle = module.register_forward_hook(hook_fn)
        try:
            self._forward(images)
        finally:
            handle.remove()

        out = activation["out"]
        if isinstance(out, tuple):
            out = out[0]

        # Pool to [B, D] if needed
        if out.dim() == 3:  # [B, tokens, D] — avg pool over tokens
            out = out.mean(dim=1)
        elif out.dim() == 4:  # [B, C, H, W] — spatial avg pool
            out = out.flatten(2).mean(dim=2)

        return out

    def list_layers(self) -> list[str]:
        """Return all named submodule paths in the model."""
        return [name for name, _ in self.model.named_modules() if name]

    def _get_submodule(self, layer: str) -> nn.Module:
        """Resolve a dot-separated layer name to a submodule."""
        module = self.model
        for attr in layer.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """Default forward pass. Override for encoders with special extraction."""
        return self.model(images)
