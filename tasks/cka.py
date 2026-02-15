"""Centered Kernel Alignment (CKA) for representation similarity analysis.

Two modes:
  - Cross-encoder: compare representations across different encoders (heatmap).
  - Cross-layer: compare representations across layers within one encoder (heatmap).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wrappers.encoder import BaseEncoder


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two feature matrices.

    Args:
        X: [N, D1] feature matrix
        Y: [N, D2] feature matrix

    Returns:
        CKA similarity in [0, 1].
    """
    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = np.linalg.norm(YtX, "fro") ** 2
    hsic_xx = np.linalg.norm(XtX, "fro")
    hsic_yy = np.linalg.norm(YtY, "fro")

    if hsic_xx * hsic_yy == 0:
        return 0.0
    return float(hsic_xy / (hsic_xx * hsic_yy))


@torch.no_grad()
def _extract_features(encoder: BaseEncoder, loader: DataLoader,
                      max_samples: int | None = None,
                      layer: str | None = None) -> np.ndarray:
    """Extract features as a numpy array [N, D]."""
    all_features = []
    collected = 0
    desc = f"Extracting ({encoder.name}" + (f", {layer})" if layer else ")")
    for images, _ in tqdm(loader, desc=desc, leave=False):
        if layer:
            features = encoder.extract_features_from_layer(images, layer)
        else:
            features = encoder.extract_features(images)
        all_features.append(features.cpu())
        collected += len(images)
        if max_samples and collected >= max_samples:
            break
    return torch.cat(all_features)[:max_samples].numpy()


def cka_cross_encoder(
    encoders: list[BaseEncoder],
    loaders: dict[str, DataLoader],
    dataset_name: str,
    out_dir: Path,
    max_samples: int = 2000,
) -> dict:
    """Compute pairwise CKA between multiple encoders. Save heatmap.

    Args:
        encoders: list of encoder instances
        loaders: dict mapping encoder.name -> DataLoader (each with its own transform)
        dataset_name: for plot title
        out_dir: output directory
        max_samples: cap on samples for CKA computation

    Returns:
        dict with 'cka_matrix', 'names', 'plot'.
    """
    names = [e.name for e in encoders]
    n = len(encoders)

    # Extract features for each encoder
    features = {}
    for enc in encoders:
        print(f"  Extracting features for {enc.name}...")
        features[enc.name] = _extract_features(enc, loaders[enc.name], max_samples=max_samples)
        print(f"    shape: {features[enc.name].shape}")

    # Compute pairwise CKA
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            score = linear_cka(features[names[i]], features[names[j]])
            cka_matrix[i, j] = score
            cka_matrix[j, i] = score
            if i != j:
                print(f"  CKA({names[i]}, {names[j]}) = {score:.4f}")

    save_path = _plot_heatmap(
        cka_matrix, names,
        title=f"Linear CKA — {dataset_name}",
        out_dir=out_dir,
        filename=f"cka_{dataset_name}_cross_encoder.png",
    )

    return {"cka_matrix": cka_matrix.tolist(), "names": names, "plot": str(save_path)}


def cka_cross_layer(
    encoder: BaseEncoder,
    loader: DataLoader,
    layers: list[str],
    dataset_name: str,
    out_dir: Path,
    max_samples: int = 2000,
) -> dict:
    """Compute pairwise CKA between layers of one encoder. Save heatmap.

    Returns:
        dict with 'cka_matrix', 'layers', 'plot'.
    """
    n = len(layers)

    # Extract features for each layer
    features = {}
    for layer in layers:
        print(f"  Extracting {encoder.name} @ {layer}...")
        features[layer] = _extract_features(encoder, loader, max_samples=max_samples, layer=layer)
        print(f"    shape: {features[layer].shape}")

    # Compute pairwise CKA
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            score = linear_cka(features[layers[i]], features[layers[j]])
            cka_matrix[i, j] = score
            cka_matrix[j, i] = score

    # Shorten layer labels for display
    short_labels = [l.split(".")[-1] if "." in l else l for l in layers]
    # If labels collide after shortening, use full names
    if len(set(short_labels)) < len(short_labels):
        short_labels = layers

    enc_tag = encoder.name.lower().replace("-", "_").replace(" ", "_")
    save_path = _plot_heatmap(
        cka_matrix, short_labels,
        title=f"Linear CKA — {encoder.name} layers — {dataset_name}",
        out_dir=out_dir,
        filename=f"cka_{dataset_name}_{enc_tag}_layers.png",
    )

    return {"cka_matrix": cka_matrix.tolist(), "layers": layers, "plot": str(save_path)}


def _plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    out_dir: Path,
    filename: str,
) -> Path:
    """Plot and save a CKA heatmap."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8 + 2), max(5, n * 0.7 + 2)))

    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, label="Linear CKA", shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] < 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = out_dir / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path
