"""
Experiment: How does progressive patch masking affect encoder performance?

Applies random patch masking at ratios [0.0, 0.1, ..., 0.9] to input images
and measures k-NN accuracy for each encoder. Outputs a line plot.

Usage:
    python -m experiments.masking --dataset cifar10 --encoder all
    python -m experiments.masking --dataset cifar10 --encoder dinov2 resnet clip
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models  # register all encoders
from models.registry import get_encoder, list_encoders
from datasets.loader import get_dataset, get_num_classes
from tasks.knn import knn_evaluate


class PatchMaskedDataset(Dataset):
    """Wraps a dataset and applies random patch masking to images."""

    def __init__(self, dataset: Dataset, mask_ratio: float, patch_size: int = 16, seed: int = 42):
        self.dataset = dataset
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.mask_ratio > 0:
            image = self._apply_mask(image, idx)
        return image, label

    def _apply_mask(self, image: torch.Tensor, idx: int) -> torch.Tensor:
        """Zero out random patches of the image tensor."""
        C, H, W = image.shape
        ph, pw = self.patch_size, self.patch_size
        nh, nw = H // ph, W // pw  # number of patches per axis
        n_patches = nh * nw
        n_mask = int(n_patches * self.mask_ratio)

        # Deterministic per-image seed so train/test masking is consistent across runs
        gen = torch.Generator().manual_seed(self.seed + idx)
        mask_indices = torch.randperm(n_patches, generator=gen)[:n_mask]

        masked = image.clone()
        for idx in mask_indices:
            row, col = divmod(idx.item(), nw)
            masked[:, row * ph : (row + 1) * ph, col * pw : (col + 1) * pw] = 0.0
        return masked


def run_experiment(encoder_names: list[str], args):
    mask_ratios = [round(r * 0.1, 1) for r in range(10)]  # 0.0 .. 0.9
    results = {}  # encoder_name -> list of accuracies

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

        # Load base datasets once (with the encoder's transform applied)
        train_ds = get_dataset(args.dataset, train=True, transform=transform, data_root=args.data_root)
        test_ds = get_dataset(args.dataset, train=False, transform=transform, data_root=args.data_root)
        num_classes = get_num_classes(args.dataset)

        accs = []
        for ratio in mask_ratios:
            print(f"  mask_ratio={ratio:.1f} ... ", end="", flush=True)

            masked_train = PatchMaskedDataset(train_ds, mask_ratio=ratio, patch_size=args.patch_size)
            masked_test = PatchMaskedDataset(test_ds, mask_ratio=ratio, patch_size=args.patch_size)

            train_loader = DataLoader(
                masked_train, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            test_loader = DataLoader(
                masked_test, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )

            result = knn_evaluate(encoder, train_loader, test_loader, num_classes=num_classes)
            acc = result["accuracy"]
            accs.append(acc)
            print(f"acc={acc:.4f}")

        results[encoder.name] = accs

    return mask_ratios, results


def plot_results(mask_ratios, results, dataset_name, save_path):
    plt.figure(figsize=(10, 6))

    for enc_name, accs in results.items():
        plt.plot(mask_ratios, accs, marker="o", linewidth=2, markersize=6, label=enc_name)

    plt.xlabel("Mask Ratio", fontsize=13)
    plt.ylabel("k-NN Accuracy", fontsize=13)
    plt.title(f"Progressive Patch Masking — {dataset_name}", fontsize=14)
    plt.xticks(mask_ratios)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Progressive masking experiment")
    parser.add_argument("--encoder", nargs="+", default=["all"],
                        help="Encoder name(s) or 'all'")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Patch size for masking (pixels)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output plot path (default: experiments/masking_{dataset}.png)")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    if "all" in args.encoder:
        encoder_names = list_encoders()
    else:
        encoder_names = args.encoder

    mask_ratios, results = run_experiment(encoder_names, args)

    if not results:
        print("No results to plot.")
        return

    out_dir = Path(__file__).resolve().parent.parent / "output" / "masking"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output or str(out_dir / f"masking_{args.dataset}_{'_'.join(args.encoder)}.png")
    plot_results(mask_ratios, results, args.dataset, save_path)


if __name__ == "__main__":
    main()
