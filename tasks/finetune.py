import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from wrappers.encoder import BaseEncoder


class FineTuneModel(nn.Module):
    """Wraps a BaseEncoder with a linear classification head for end-to-end training."""

    def __init__(self, encoder: BaseEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder._forward(x)
        return self.head(features)


def _build_train_transform(eval_transform, resolution: int = 224) -> transforms.Compose:
    """Build a training transform with augmentation, matching eval normalization."""
    # Extract Normalize params from the eval transform
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet defaults
    for t in eval_transform.transforms:
        if isinstance(t, transforms.Normalize):
            mean = list(t.mean)
            std = list(t.std)
            break

    # Extract resolution from eval transform
    for t in eval_transform.transforms:
        if isinstance(t, transforms.Resize):
            size = t.size
            resolution = size if isinstance(size, int) else size[0]
            break
        elif isinstance(t, transforms.CenterCrop):
            size = t.size
            resolution = size if isinstance(size, int) else size[0]
            break

    return transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def finetune_evaluate(
    encoder: BaseEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    device: str | None = None,
) -> dict:
    """Fine-tune encoder end-to-end with a linear head on top.

    Returns dict with 'accuracy', 'best_epoch', 'epochs'.
    """
    device = device or encoder.device

    # Build the fine-tune model (encoder + head)
    model = FineTuneModel(encoder, num_classes).to(device)
    model.encoder.model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Fine-tuning", leave=False):
        # --- Train ---
        model.train()
        model.encoder.model.train()
        total_loss = 0.0
        n_batches = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # --- Eval ---
        model.eval()
        model.encoder.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        acc = correct / total
        avg_loss = total_loss / n_batches
        tqdm.write(f"  Epoch {epoch:>3d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

    return {"accuracy": best_acc, "best_epoch": best_epoch, "epochs": epochs}
