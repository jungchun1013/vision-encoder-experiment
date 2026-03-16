"""t-SNE / PCA visualization of clock representations.

Single plot (or multi-layer subplots) showing how an encoder organises
720 synthetic clock images (12 hours x 60 minutes).

HSV colouring scheme
--------------------
* Hue        — hour value (0-11 → 0–1, circular)
* Saturation — minute value (0-59 → 1.0–0.3, vivid→muted)
* Value      — always 1.0
"""

import csv
from colorsys import hsv_to_rgb
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.decomposition import PCA

from tasks.tsne import _run_tsne
from wrappers.encoder import BaseEncoder

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "clock"


# ── dataset ────────────────────────────────────────────────────────────
class ClockDataset(Dataset):
    """Load clock images with hour / minute metadata."""

    def __init__(self, root: str | Path = DATA_ROOT, transform=None):
        self.root = Path(root)
        self.transform = transform

        meta_path = self.root / "metadata.csv"
        self.samples: list[dict] = []
        with open(meta_path, newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "fname": row["rgb_fname"],
                    "hour": int(row["hour"]),
                    "minute": int(row["minute"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        img = Image.open(self.root / "image" / meta["fname"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, meta


# ── helpers ────────────────────────────────────────────────────────────
def _run_pca(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components).fit_transform(features)


def _reduce(features, reduction, perplexity):
    if reduction == "pca":
        return _run_pca(features, n_components=2)
    return _run_tsne(features, perplexity=perplexity, n_components=2)


def _make_colors(hours, minutes):
    """HSV colours: hue=hour (circular), saturation=minute (vivid→muted)."""
    sat_max, sat_min = 1.0, 0.3
    return np.array([
        hsv_to_rgb(h / 12.0, sat_max - (m / 59.0) * (sat_max - sat_min), 1.0)
        for h, m in zip(hours, minutes)
    ])


def _minute_edges(hours, minutes):
    """Build (idx_a, idx_b) pairs connecting consecutive minutes within each hour.

    For each hour group, connects minute m to minute m+1 (wrapping 59→0).
    """
    # Build lookup: (hour, minute) → index
    lookup: dict[tuple[int, int], int] = {}
    for i, (h, m) in enumerate(zip(hours, minutes)):
        lookup[(int(h), int(m))] = i

    edges = []
    for (h, m), idx_a in lookup.items():
        next_m = (m + 1) % 60
        idx_b = lookup.get((h, next_m))
        if idx_b is not None:
            print(f"({h}, {m}) -> ({h}, {next_m})")
            edges.append((idx_a, idx_b))
        else:
            print(f"({h}, {m}) -> No next minute")
    
    return edges


def _style_ax(ax, emb, colors, title, hours, minutes):
    """Draw edges, scatter, and legend on a single axis."""
    ax.set_facecolor("#1a1a1a")

    # Draw lines between consecutive minutes (within each hour)
    edges = _minute_edges(hours, minutes)
    segments = []
    edge_colors = []
    for a, b in edges:
        segments.append([emb[a], emb[b]])
        edge_colors.append((colors[a] + colors[b]) / 2)
    lc = LineCollection(segments, colors=edge_colors,
                        linewidths=1.0, alpha=0.4, zorder=1)
    ax.add_collection(lc)

    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=14, alpha=0.9,
               edgecolors="none", zorder=2)

    # compact legend: sample hours as hue swatches (full saturation)
    for h in [0, 3, 6, 9]:
        c = hsv_to_rgb(h / 12.0, 1.0, 1.0)
        label_h = 12 if h == 0 else h
        ax.scatter([], [], c=[c], s=25, label=f"{label_h}:00")
    ax.legend(loc="lower center", ncol=4, fontsize=7,
              frameon=False, labelcolor="white",
              bbox_to_anchor=(0.5, -0.06))

    ax.set_title(title, fontsize=12, fontweight="bold", color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── main task ──────────────────────────────────────────────────────────
@torch.no_grad()
def clock_tsne_evaluate(
    encoder: BaseEncoder,
    out_dir: Path,
    reduction: str = "tsne",
    perplexity: float = 30,
    batch_size: int = 64,
    layers: list[str] | None = None,
) -> dict:
    """Extract features from clock images, reduce dims, plot symbol map.

    If *layers* is provided, produces one subplot per layer.
    Otherwise uses the final encoder output.
    """
    transform = encoder.get_transform()
    dataset = ClockDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Decide which representations to extract
    layer_list = layers or [None]  # None = final output

    # Extract features for every requested layer in one pass per batch
    feat_dict: dict[str | None, list] = {l: [] for l in layer_list}
    hours, minutes = [], []

    for images, meta_batch in tqdm(loader, desc=f"Extracting ({encoder.name})", leave=False):
        for layer in layer_list:
            if layer is None:
                feats = encoder.extract_features(images)
            else:
                feats = encoder.extract_features_from_layer(images, layer)
            feat_dict[layer].append(feats.cpu())
        hours.extend(meta_batch["hour"])
        minutes.extend(meta_batch["minute"])

    hours = np.array([h.item() if isinstance(h, torch.Tensor) else int(h) for h in hours])
    minutes = np.array([m.item() if isinstance(m, torch.Tensor) else int(m) for m in minutes])
    n = len(hours)
    colors = _make_colors(hours, minutes)
    red_label = reduction.upper()

    # Build figure — one subplot per layer
    n_plots = len(layer_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
    fig.patch.set_facecolor("#1a1a1a")
    if n_plots == 1:
        axes = [axes]

    for ax, layer in zip(axes, layer_list):
        features = torch.cat(feat_dict[layer]).numpy()
        label = layer if layer else "output"
        print(f"  [{label}] features: {features.shape} — running {red_label}...")
        emb = _reduce(features, reduction, perplexity)
        title = f"{label}  ({features.shape[1]}d)"
        _style_ax(ax, emb, colors, title, hours, minutes)

    suptitle = f"{encoder.name} — Clock {red_label}  ({n} images)"
    suptitle += "\nhue = hour  |  saturation = minute (1.0→0.3)"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", color="white", y=1.03)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_tag = encoder.name.lower().replace("-", "_").replace(" ", "_")
    layer_tag = "_layers" if layers else ""
    save_path = out_dir / f"clock_{reduction}_{enc_tag}{layer_tag}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {"encoder": encoder.name, "plot": str(save_path), "samples": n}
