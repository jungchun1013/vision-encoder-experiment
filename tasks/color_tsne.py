"""t-SNE visualization of how encoders represent pure solid colors."""

from colorsys import hsv_to_rgb
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.decomposition import PCA

from tasks.tsne import _run_tsne
from wrappers.encoder import BaseEncoder


def _run_pca(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Run PCA on features, return embeddings of shape [N, n_components]."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

# Color names with their canonical RGB values (for label text color on plot)
COLOR_LABELS = {
    "red":     (255,   0,   0),
    "orange":  (255, 165,   0),
    "yellow":  (255, 255,   0),
    "green":   (  0, 128,   0),
    "blue":    (  0,   0, 255),
    "purple":  (128,   0, 128),
    "white":   (255, 255, 255),
    "black":   (  0,   0,   0),
    "brown":   (139,  69,  19),
    "pink":    (255, 192, 203),
    "gray":    (128, 128, 128),
    "cyan":    (  0, 255, 255),
}


def _generate_solid_colors_rgb(steps: int = 8):
    """Uniform grid in RGB space."""
    levels = np.linspace(0, 255, steps, dtype=int)
    r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
    return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1).astype(np.uint8)


def _generate_solid_colors_hsv(steps: int = 8):
    """Uniform grid in HSV space, converted to RGB."""
    h_levels = np.linspace(0, 1, steps, endpoint=False)  # hue is circular
    s_levels = np.linspace(0, 1, steps)
    v_levels = np.linspace(0, 1, steps)
    h, s, v = np.meshgrid(h_levels, s_levels, v_levels, indexing="ij")
    h, s, v = h.ravel(), s.ravel(), v.ravel()
    rgb = np.array([hsv_to_rgb(hi, si, vi) for hi, si, vi in zip(h, s, v)])
    return (rgb * 255).astype(np.uint8)


def _grid_edges(steps: int, wrap_first: bool = False):
    """Return list of (idx_a, idx_b) for grid-adjacent colors.

    Args:
        steps: grid resolution per axis (steps³ total points)
        wrap_first: if True, wrap the first axis (hue in HSV is circular)
    """
    edges = []
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                idx = i * steps * steps + j * steps + k
                # +1 along each axis
                if i + 1 < steps:
                    edges.append((idx, (i + 1) * steps * steps + j * steps + k))
                elif wrap_first:
                    edges.append((idx, 0 * steps * steps + j * steps + k))
                if j + 1 < steps:
                    edges.append((idx, i * steps * steps + (j + 1) * steps + k))
                if k + 1 < steps:
                    edges.append((idx, i * steps * steps + j * steps + (k + 1)))
    return edges


@torch.no_grad()
def _encode_color_texts(model, device):
    """Encode color name strings with CLIP's text encoder.

    Returns (text_features, names, rgb_for_labels) or None if not a CLIP model.
    """
    if not hasattr(model, "encode_text"):
        return None

    try:
        import open_clip
    except ImportError:
        return None

    names = list(COLOR_LABELS.keys())
    prompts = [f"a photo of something {name}" for name in names]

    # open_clip tokenizer — works for any open_clip model
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(tokens)  # [K, D]
    text_features = text_features.float().cpu().numpy()

    rgb_for_labels = np.array([COLOR_LABELS[n] for n in names], dtype=np.uint8)
    return text_features, names, rgb_for_labels


@torch.no_grad()
def color_tsne_evaluate(
    encoder: BaseEncoder,
    out_dir: Path,
    color_space: str = "rgb",
    reduction: str = "tsne",
    perplexity: float = 30,
    batch_size: int = 256,
) -> dict:
    """Generate solid-color images, extract features, reduce dims, and plot."""
    transform = encoder.get_transform()

    # Determine image resolution from the transform
    dummy = Image.new("RGB", (256, 256), (128, 128, 128))
    dummy_t = transform(dummy)
    resolution = dummy_t.shape[-1]

    # Generate solid-color images in chosen color space
    if color_space == "hsv":
        rgb_values = _generate_solid_colors_hsv(steps=8)
    else:
        rgb_values = _generate_solid_colors_rgb(steps=8)
    n_colors = len(rgb_values)
    print(f"  Generated {n_colors} solid-color images ({color_space.upper()}) at {resolution}x{resolution}")

    # Create transformed tensor dataset
    tensors = []
    for rgb in tqdm(rgb_values, desc="Preparing images", leave=False):
        img = Image.new("RGB", (resolution, resolution), tuple(rgb))
        tensors.append(transform(img))
    images_tensor = torch.stack(tensors)

    loader = DataLoader(
        TensorDataset(images_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Extract visual features
    all_features = []
    for (batch,) in tqdm(loader, desc=f"Extracting ({encoder.name})", leave=False):
        features = encoder.extract_features(batch)
        all_features.append(features.cpu())
    vis_features = torch.cat(all_features).numpy()
    print(f"  Visual features: {vis_features.shape}")

    # Try to get text features for color names (CLIP-family only)
    text_result = _encode_color_texts(encoder.model, encoder.device)
    has_text = text_result is not None

    # Reduce to 2D and 3D
    red_label = reduction.upper()  # "TSNE" or "PCA"
    plots = []
    for ndim in (2, 3):
        if has_text:
            text_features, color_names, label_rgbs = text_result
            if ndim == 2:
                print(f"  Text features: {text_features.shape} ({len(color_names)} color names)")
            combined = np.concatenate([vis_features, text_features], axis=0)
            print(f"  Running {ndim}D {red_label} (visual + text)...")
            if reduction == "pca":
                all_embeddings = _run_pca(combined, n_components=ndim)
            else:
                all_embeddings = _run_tsne(combined, perplexity=perplexity,
                                           n_components=ndim)
            vis_emb = all_embeddings[:n_colors]
            text_emb = all_embeddings[n_colors:]
        else:
            print(f"  Running {ndim}D {red_label}...")
            if reduction == "pca":
                vis_emb = _run_pca(vis_features, n_components=ndim)
            else:
                vis_emb = _run_tsne(vis_features, perplexity=perplexity,
                                    n_components=ndim)

        colors_norm = rgb_values.astype(np.float32) / 255.0
        edges = _grid_edges(8, wrap_first=(color_space == "hsv"))

        if ndim == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        # Draw edges between grid-adjacent colors
        if ndim == 2:
            segments = []
            edge_colors_list = []
            for a, b in edges:
                segments.append([vis_emb[a], vis_emb[b]])
                edge_colors_list.append((colors_norm[a] + colors_norm[b]) / 2)
            lc = LineCollection(segments, colors=edge_colors_list,
                                linewidths=1, alpha=0.25, zorder=1)
            ax.add_collection(lc)
        else:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            segments = []
            edge_colors_list = []
            for a, b in edges:
                segments.append([vis_emb[a], vis_emb[b]])
                edge_colors_list.append(
                    np.append((colors_norm[a] + colors_norm[b]) / 2, 0.25))
            lc = Line3DCollection(segments, colors=edge_colors_list,
                                  linewidths=0.8)
            ax.add_collection3d(lc)

        # Visual feature scatter
        if ndim == 2:
            ax.scatter(vis_emb[:, 0], vis_emb[:, 1],
                       c=colors_norm, s=10, alpha=0.95,
                       edgecolors="none", zorder=2)
        else:
            ax.scatter(vis_emb[:, 0], vis_emb[:, 1], vis_emb[:, 2],
                       c=colors_norm, s=10, alpha=0.95, edgecolors="none")

        # Overlay text labels for CLIP-family
        if has_text:
            for i, name in enumerate(color_names):
                rgb_norm = label_rgbs[i].astype(np.float32) / 255.0
                if ndim == 2:
                    ax.annotate(
                        name.upper(),
                        (text_emb[i, 0], text_emb[i, 1]),
                        fontsize=10, fontweight="bold",
                        ha="center", va="center", color=rgb_norm,
                        path_effects=[pe.withStroke(linewidth=3,
                                                    foreground="black")],
                    )
                    ax.scatter(text_emb[i, 0], text_emb[i, 1],
                               marker="D", s=40, c=[rgb_norm],
                               edgecolors="white", linewidths=0.5, zorder=5)
                else:
                    ax.text(text_emb[i, 0], text_emb[i, 1], text_emb[i, 2],
                            f"  {name.upper()}",
                            fontsize=9, fontweight="bold", color=rgb_norm,
                            path_effects=[pe.withStroke(linewidth=3,
                                                        foreground="black")])
                    ax.scatter(text_emb[i, 0], text_emb[i, 1], text_emb[i, 2],
                               marker="D", s=40, c=[rgb_norm],
                               edgecolors="white", linewidths=0.5)

        title_suffix = " + language labels" if has_text else ""
        cs_label = color_space.upper()
        dim_label = "2D" if ndim == 2 else "3D"
        ax.set_title(
            f"{encoder.name} — Color {red_label} {dim_label} [{cs_label}] "
            f"({n_colors} colors{title_suffix})",
            fontsize=14, fontweight="bold", color="white",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if ndim == 2:
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("#333333")
            ax.yaxis.pane.set_edgecolor("#333333")
            ax.zaxis.pane.set_edgecolor("#333333")
        fig.tight_layout()

        enc_tag = encoder.name.lower().replace("-", "_").replace(" ", "_")
        suffix = "2d" if ndim == 2 else "3d"
        save_path = out_dir / f"color_{reduction}_{color_space}_{enc_tag}_{suffix}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        plots.append(str(save_path))
        print(f"  Saved: {save_path}")

    return {"encoder": encoder.name, "plots": plots, "samples": n_colors,
            "color_space": color_space}
