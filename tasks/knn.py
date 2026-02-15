import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


def knn_evaluate(
    encoder: BaseEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    k: int = 20,
    num_classes: int | None = None,
    layer: str | None = None,
) -> dict:
    """k-NN evaluation with cosine similarity and weighted voting.

    Returns dict with 'accuracy' and 'k'.
    """
    train_features, train_labels = _extract_all(encoder, train_loader, layer=layer)
    test_features, test_labels = _extract_all(encoder, test_loader, layer=layer)

    # L2-normalize for cosine similarity
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    if num_classes is None:
        num_classes = int(train_labels.max().item()) + 1

    # Compute k-NN in chunks to avoid OOM
    chunk_size = 256
    correct = 0
    total = 0

    for i in range(0, len(test_features), chunk_size):
        chunk = test_features[i : i + chunk_size]
        # Cosine similarity = dot product of normalized vectors
        sim = chunk @ train_features.T  # [chunk_size, N_train]
        topk_sim, topk_idx = sim.topk(k, dim=1)  # [chunk_size, k]

        # Weighted voting: weight by similarity
        topk_labels = train_labels[topk_idx]  # [chunk_size, k]
        votes = torch.zeros(len(chunk), num_classes)
        for c in range(num_classes):
            mask = (topk_labels == c).float()
            votes[:, c] = (mask * topk_sim).sum(dim=1)

        preds = votes.argmax(dim=1)
        correct += (preds == test_labels[i : i + chunk_size]).sum().item()
        total += len(chunk)

    accuracy = correct / total
    return {"accuracy": accuracy, "k": k}
