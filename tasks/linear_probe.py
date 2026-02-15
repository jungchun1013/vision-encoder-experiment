import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from wrappers.encoder import BaseEncoder


@torch.no_grad()
def _extract_all(encoder: BaseEncoder, loader: DataLoader,
                 layer: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features and labels for an entire dataloader."""
    all_features, all_labels = [], []
    desc = f"Extracting ({encoder.name}" + (f", {layer})" if layer else ")")
    for images, labels in tqdm(loader, desc=desc, leave=False):
        if layer:
            features = encoder.extract_features_from_layer(images, layer)
        else:
            features = encoder.extract_features(images)
        all_features.append(features.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def linear_probe_evaluate(
    encoder: BaseEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    epochs: int = 50,
    lr: float = 0.1,
    batch_size: int = 256,
    device: str | None = None,
    layer: str | None = None,
) -> dict:
    """Linear probe: freeze encoder, train nn.Linear on extracted features.

    Returns dict with 'accuracy', 'epochs', 'best_epoch'.
    """
    device = device or encoder.device

    # Pre-extract all features (once)
    layer_info = f" @ {layer}" if layer else ""
    print(f"Pre-extracting features for linear probe{layer_info}...")
    train_features, train_labels = _extract_all(encoder, train_loader, layer=layer)
    test_features, test_labels = _extract_all(encoder, test_loader, layer=layer)

    feature_dim = train_features.shape[1]

    # Build feature dataloaders
    train_ds = TensorDataset(train_features, train_labels)
    test_ds = TensorDataset(test_features, test_labels)
    feat_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    feat_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Linear head
    head = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Linear probe", leave=False):
        # Train
        head.train()
        for feats, labels in feat_train:
            feats, labels = feats.to(device), labels.to(device)
            logits = head(feats)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Eval
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feats, labels in feat_test:
                feats, labels = feats.to(device), labels.to(device)
                preds = head(feats).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

    return {"accuracy": best_acc, "epochs": epochs, "best_epoch": best_epoch}
