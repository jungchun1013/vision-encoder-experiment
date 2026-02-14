"""
Experiment: Visualize encoder representations using t-SNE.

Extracts features from the test set, reduces to 2D with t-SNE,
and plots a scatter colored by class label. Supports multiple encoders
side-by-side in a single figure.

Usage:
    python -m experiments.tsne --encoder dinov2 --dataset cifar10
    python -m experiments.tsne --encoder dinov2 resnet clip --dataset cifar10
    python -m experiments.tsne --encoder all --dataset cifar10 --max-samples 2000

    # Extract from a specific layer:
    python -m experiments.tsne --encoder dinov2 --layer blocks.5 --dataset cifar10
    python -m experiments.tsne --encoder resnet --layer layer3 --dataset cifar10

    # List available layers for an encoder:
    python -m experiments.tsne --encoder dinov2 --list-layers
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models  # register all encoders
from models.registry import get_encoder, list_encoders
from datasets.loader import get_dataset, get_num_classes

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def extract_features(encoder, loader, max_samples: int | None = None, layer: str | None = None):
    """Extract features and labels from a dataloader.

    If layer is specified, extracts activations from that layer using a forward hook.
    Otherwise uses the encoder's default output.
    """
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


def run_tsne(features: np.ndarray, perplexity: float = 30, seed: int = 42) -> np.ndarray:
    """Run t-SNE on features, return 2D embeddings."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    return tsne.fit_transform(features)


def get_class_names(dataset_name: str, num_classes: int) -> list[str] | None:
    """Return class names if known, else None."""
    if dataset_name == "cifar10":
        return CIFAR10_CLASSES
    return None


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of encoder representations")
    parser.add_argument("--encoder", nargs="+", default=["dinov2"],
                        help="Encoder name(s) or 'all'")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples for t-SNE (more = slower)")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--layer", type=str, default=None,
                        help="Extract from a specific layer (e.g. 'blocks.5', 'layer3')")
    parser.add_argument("--list-layers", action="store_true",
                        help="Print available layer names for the encoder(s) and exit")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    if "all" in args.encoder:
        encoder_names = list_encoders()
    else:
        encoder_names = args.encoder

    # --list-layers mode: print layer names and exit
    if args.list_layers:
        for enc_name in encoder_names:
            try:
                encoder = get_encoder(enc_name, device=args.device)
                layers = encoder.list_layers()
                print(f"\n{encoder.name} — {len(layers)} layers:")
                for l in layers:
                    print(f"  {l}")
            except NotImplementedError as e:
                print(f"\n  [SKIP] {enc_name}: {e}")
        return

    num_classes = get_num_classes(args.dataset)
    class_names = get_class_names(args.dataset, num_classes)

    # Collect results: list of (encoder_name, embeddings_2d, labels)
    plots = []

    for enc_name in encoder_names:
        print(f"\n{'='*60}")
        print(f"Encoder: {enc_name}")
        print(f"{'='*60}")

        try:
            encoder = get_encoder(enc_name, device=args.device)
            transform = encoder.get_transform()
        except NotImplementedError as e:
            print(f"  [SKIP] {e}")
            continue

        is_train = args.split == "train"
        dataset = get_dataset(args.dataset, train=is_train, transform=transform, data_root=args.data_root)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        layer_info = f" @ {args.layer}" if args.layer else ""
        features, labels = extract_features(encoder, loader, max_samples=args.max_samples, layer=args.layer)
        print(f"  Features{layer_info}: {features.shape}")

        print(f"  Running t-SNE...")
        embeddings = run_tsne(features, perplexity=args.perplexity)
        plots.append((encoder.name, embeddings, labels))

    if not plots:
        print("No results to plot.")
        return

    out_dir = Path(__file__).resolve().parent.parent / "output" / "tsne"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.cm.get_cmap("tab10" if num_classes <= 10 else "tab20", num_classes)
    layer_suffix = f" @ {args.layer}" if args.layer else ""
    layer_tag = f"_{args.layer.replace('.', '_')}" if args.layer else ""

    # Legend handles (shared across all plots)
    handles = []
    for c in range(num_classes):
        label = class_names[c] if class_names else str(c)
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=cmap(c), markersize=8, label=label))

    # Save each encoder as its own figure
    for enc_name, emb, labels in plots:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            emb[:, 0], emb[:, 1],
            c=labels, cmap=cmap, s=5, alpha=0.6, edgecolors="none",
        )
        ax.set_title(f"{enc_name} — {args.dataset}{layer_suffix}",
                     fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(handles=handles, loc="lower center", ncol=min(num_classes, 5),
                  fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()

        enc_tag = enc_name.lower().replace("-", "_").replace(" ", "_")
        save_path = str(out_dir / f"tsne_{args.dataset}{layer_tag}_{enc_tag}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
