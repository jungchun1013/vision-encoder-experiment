from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as tvd

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class FragmentDataset(Dataset):
    """Load a specific image variant from each subfolder.

    Parameters
    ----------
    root : str
        Parent of the fragment directory (e.g. ``data/``).
    folder : str
        Subdirectory name under *root* (``"fragment"`` or ``"fragment_v2"``).
    filename : str
        Which PNG to load from each subfolder
        (``"original_original.png"``, ``"original.png"``, ``"gray.png"``, …).
    """

    def __init__(self, root: str, train: bool, transform=None,
                 folder: str = "fragment", filename: str = "original_original.png"):
        frag_dir = Path(root) / folder
        self.filename = filename
        self.samples: list[Path] = sorted(
            p for p in frag_dir.iterdir()
            if p.is_dir() and (p / filename).exists()
        )
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx] / self.filename
        img = Image.open(img_path).convert("RGB")
        label = idx
        if self.transform:
            img = self.transform(img)
        return img, label


_DATASET_BUILDERS = {
    "cifar10": lambda root, train, transform: tvd.CIFAR10(
        root=root, train=train, transform=transform, download=True
    ),
    "stl10": lambda root, train, transform: tvd.STL10(
        root=root, split="train" if train else "test", transform=transform, download=True
    ),
    "flowers102": lambda root, train, transform: tvd.Flowers102(
        root=root, split="train" if train else "test", transform=transform, download=True
    ),
    "food101": lambda root, train, transform: tvd.Food101(
        root=root, split="train" if train else "test", transform=transform, download=True
    ),
    "oxfordpets": lambda root, train, transform: tvd.OxfordIIITPet(
        root=root, split="trainval" if train else "test", transform=transform, download=True
    ),
    "imagenet1k": lambda root, train, transform: tvd.ImageNet(
        root=str(Path(root) / "imagenet"), split="train" if train else "val", transform=transform,
    ),
    "fragmentoriginal": lambda root, train, transform: FragmentDataset(
        root=root, train=train, transform=transform,
        folder="fragment_v2", filename="original.png",
    ),
    "fragmentlined": lambda root, train, transform: FragmentDataset(
        root=root, train=train, transform=transform,
        folder="fragment_v2", filename="lined.png",
    ),
    "fragmentgray": lambda root, train, transform: FragmentDataset(
        root=root, train=train, transform=transform,
        folder="fragment_v2", filename="gray.png",
    ),
}


def get_dataset(name: str, train: bool = True, transform=None, data_root: str | None = None):
    """Load a dataset by name."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    key = name.lower().replace("-", "").replace("_", "")
    if key not in _DATASET_BUILDERS:
        available = ", ".join(sorted(_DATASET_BUILDERS))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATASET_BUILDERS[key](str(root), train, transform)


def get_num_classes(name: str) -> int:
    """Return the number of classes for a dataset."""
    counts = {
        "cifar10": 10,
        "stl10": 10,
        "flowers102": 102,
        "food101": 101,
        "oxfordpets": 37,
        "imagenet1k": 1000,
        "fragmentoriginal": 260,
        "fragmentlined": 260,
        "fragmentgray": 260,
    }
    key = name.lower().replace("-", "").replace("_", "")
    if key not in counts:
        raise ValueError(f"Unknown dataset '{name}'")
    return counts[key]


def get_loaders(
    name: str,
    transform,
    batch_size: int = 64,
    data_root: str | None = None,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for a dataset."""
    train_ds = get_dataset(name, train=True, transform=transform, data_root=data_root)
    test_ds = get_dataset(name, train=False, transform=transform, data_root=data_root)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
