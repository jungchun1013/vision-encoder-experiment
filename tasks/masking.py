"""Progressive patch masking experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wrappers.encoder import BaseEncoder
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
        nh, nw = H // ph, W // pw
        n_patches = nh * nw
        n_mask = int(n_patches * self.mask_ratio)

        gen = torch.Generator().manual_seed(self.seed + idx)
        mask_indices = torch.randperm(n_patches, generator=gen)[:n_mask]

        masked = image.clone()
        for i in mask_indices:
            row, col = divmod(i.item(), nw)
            masked[:, row * ph : (row + 1) * ph, col * pw : (col + 1) * pw] = 0.0
        return masked


def masking_evaluate(
    encoder: BaseEncoder,
    train_dataset: Dataset,
    test_dataset: Dataset,
    dataset_name: str,
    num_classes: int,
    out_dir: Path,
    batch_size: int = 64,
    patch_size: int = 16,
    num_workers: int = 4,
) -> dict:
    """Run progressive masking and save a line plot. Returns metadata dict."""
    mask_ratios = [round(r * 0.1, 1) for r in range(10)]
    accs = []

    for ratio in mask_ratios:
        print(f"  mask_ratio={ratio:.1f} ... ", end="", flush=True)

        masked_train = PatchMaskedDataset(train_dataset, mask_ratio=ratio, patch_size=patch_size)
        masked_test = PatchMaskedDataset(test_dataset, mask_ratio=ratio, patch_size=patch_size)

        train_loader = DataLoader(
            masked_train, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        test_loader = DataLoader(
            masked_test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        result = knn_evaluate(encoder, train_loader, test_loader, num_classes=num_classes)
        acc = result["accuracy"]
        accs.append(acc)
        print(f"acc={acc:.4f}")

    # Plot
    enc_tag = encoder.name.lower().replace("-", "_").replace(" ", "_")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mask_ratios, accs, marker="o", linewidth=2, markersize=6, label=encoder.name)
    ax.set_xlabel("Mask Ratio", fontsize=13)
    ax.set_ylabel("k-NN Accuracy", fontsize=13)
    ax.set_title(f"Progressive Patch Masking — {encoder.name} — {dataset_name}", fontsize=14)
    ax.set_xticks(mask_ratios)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = out_dir / f"masking_{dataset_name}_{enc_tag}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {
        "encoder": encoder.name,
        "plot": str(save_path),
        "accuracies": dict(zip(mask_ratios, accs)),
    }
