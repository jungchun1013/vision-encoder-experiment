"""t-SNE visualization of encoder representations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from wrappers.encoder import BaseEncoder

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def _extract_features(encoder: BaseEncoder, loader: DataLoader,
                      max_samples: int | None = None, layer: str | None = None):
    """Extract features and labels from a dataloader."""
    all_features, all_labels = [], []
    collected = 0
    desc = f"Extracting ({encoder.name}" + (f", {layer})" if layer else ")")
    for images, labels in tqdm(loader, desc=desc, leave=False):
        if layer:
            features = encoder.extract_features_from_layer(images, layer)
        else:
            features = encoder.extract_features(images)
        all_features.append(features.cpu())
        all_labels.append(labels)
        collected += len(labels)
        if max_samples and collected >= max_samples:
            break
    features = torch.cat(all_features)[:max_samples]
    labels = torch.cat(all_labels)[:max_samples]
    return features.numpy(), labels.numpy()


def _run_tsne(features: np.ndarray, perplexity: float = 30, seed: int = 42,
              n_components: int = 2) -> np.ndarray:
    """Run t-SNE on features, return embeddings of shape [N, n_components]."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=seed, init="pca")
    return tsne.fit_transform(features)


def _get_class_names(dataset_name: str) -> list[str] | None:
    if dataset_name == "cifar10":
        return CIFAR10_CLASSES
    return None


def tsne_evaluate(
    encoder: BaseEncoder,
    test_loader: DataLoader,
    dataset_name: str,
    num_classes: int,
    out_dir: Path,
    layer: str | None = None,
    max_samples: int = 5000,
    perplexity: float = 30,
) -> dict:
    """Run t-SNE and save a scatter plot. Returns metadata dict."""
    layer_info = f" @ {layer}" if layer else ""
    features, labels = _extract_features(encoder, test_loader, max_samples=max_samples, layer=layer)
    print(f"  Features{layer_info}: {features.shape}")

    print(f"  Running t-SNE...")
    embeddings = _run_tsne(features, perplexity=perplexity)

    # Plot
    class_names = _get_class_names(dataset_name)
    cmap = plt.cm.get_cmap("tab10" if num_classes <= 10 else "tab20", num_classes)

    handles = []
    for c in range(num_classes):
        label = class_names[c] if class_names else str(c)
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=cmap(c), markersize=8, label=label))

    layer_suffix = f" @ {layer}" if layer else ""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap=cmap, s=5, alpha=0.6, edgecolors="none",
    )
    ax.set_title(f"{encoder.name} — {dataset_name}{layer_suffix}",
                 fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=handles, loc="lower center", ncol=min(num_classes, 5),
              fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()

    enc_tag = encoder.name.lower().replace("-", "_").replace(" ", "_")
    layer_tag = f"_{layer.replace('.', '_')}" if layer else ""
    save_path = out_dir / f"tsne_{dataset_name}{layer_tag}_{enc_tag}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {"encoder": encoder.name, "plot": str(save_path), "samples": len(labels)}
