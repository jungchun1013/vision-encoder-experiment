"""Reconstruction task: evaluate and visualize encoder reconstruction quality.

- MAE: actual pixel reconstruction via encoder+decoder (ViTMAEForPreTraining)
  Metrics: MSE, PSNR, SSIM (on masked patches only + full image)
- Other encoders: nearest-neighbor retrieval in feature space
  Metrics: Precision@k (class match), mean cosine similarity
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from wrappers.encoder import BaseEncoder


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean squared error between two image tensors."""
    return F.mse_loss(x, y).item()


def _psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio."""
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


def _ssim_single(x: np.ndarray, y: np.ndarray) -> float:
    """Structural similarity for a single [H, W, C] image pair (simplified)."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(num / den)


def _ssim_batch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean SSIM over a batch of [B, C, H, W] images."""
    scores = []
    for i in range(x.shape[0]):
        xi = x[i].permute(1, 2, 0).cpu().numpy()
        yi = y[i].permute(1, 2, 0).cpu().numpy()
        scores.append(_ssim_single(xi, yi))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# MAE reconstruction
# ---------------------------------------------------------------------------

def _load_mae_for_reconstruction(device: str):
    """Load ViTMAEForPreTraining with default mask_ratio (0.75)."""
    from transformers import ViTMAEForPreTraining
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    model = model.to(device).eval()
    return model


def _unpatchify(x: torch.Tensor, patch_size: int = 16, img_size: int = 224) -> torch.Tensor:
    """Convert patch predictions [B, N, patch_size**2 * 3] back to images [B, 3, H, W]."""
    h = w = img_size // patch_size
    B, N, D = x.shape
    x = x.reshape(B, h, w, patch_size, patch_size, 3)
    x = x.permute(0, 5, 1, 3, 2, 4)  # [B, 3, h, p, w, p]
    x = x.reshape(B, 3, img_size, img_size)
    return x


def _patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """Convert images [B, 3, H, W] to patches [B, N, patch_size**2 * 3]."""
    B, C, H, W = imgs.shape
    h, w = H // patch_size, W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)  # [B, h, w, p, p, C]
    x = x.reshape(B, h * w, patch_size ** 2 * C)
    return x


@torch.no_grad()
def _mae_reconstruct(images: torch.Tensor, model, mask_ratio: float = 0.75):
    """Run MAE reconstruction. Returns (reconstructed, masked_img, composite, mask)."""
    old_ratio = model.config.mask_ratio
    model.config.mask_ratio = mask_ratio
    model.vit.embeddings.config.mask_ratio = mask_ratio

    output = model(pixel_values=images)

    model.config.mask_ratio = old_ratio
    model.vit.embeddings.config.mask_ratio = old_ratio

    pred = output.logits   # [B, num_patches, patch_size**2 * 3]
    mask = output.mask      # [B, num_patches] — 1 = masked, 0 = visible

    recon = _unpatchify(pred)

    patches = _patchify(images)
    masked_patches = patches * (1 - mask.unsqueeze(-1).float())
    masked_img = _unpatchify(masked_patches)

    composite_patches = patches * (1 - mask.unsqueeze(-1).float()) + pred * mask.unsqueeze(-1).float()
    composite = _unpatchify(composite_patches)

    return recon, masked_img, composite, mask


def _denormalize(img: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Denormalize and convert to [H, W, C] numpy in [0, 1]."""
    img = img.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def _denormalize_tensor(imgs: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """Denormalize a batch [B, C, H, W] to [0, 1] range (stays on same device)."""
    out = imgs.clone()
    for c in range(3):
        out[:, c] = out[:, c] * std[c] + mean[c]
    return out.clamp(0, 1)


def mae_reconstruction_evaluate(
    encoder: BaseEncoder,
    test_loader: DataLoader,
    dataset_name: str,
    out_dir: Path,
    num_images: int = 8,
    mask_ratio: float = 0.75,
) -> dict:
    """MAE reconstruction: compute metrics and visualize.

    Metrics (computed over multiple batches):
      - MSE (full image, masked patches only)
      - PSNR (full, masked only)
      - SSIM (full)
    """
    model = _load_mae_for_reconstruction(encoder.device)

    # Compute metrics over multiple batches
    all_mse_full, all_mse_masked = [], []
    all_psnr_full, all_psnr_masked = [], []
    all_ssim = []
    eval_batches = 10
    vis_images = vis_masked = vis_recon = vis_composite = None

    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx >= eval_batches:
            break
        images = images.to(encoder.device)
        recon, masked_img, composite, mask = _mae_reconstruct(images, model, mask_ratio)

        # Save first batch for visualization
        if batch_idx == 0:
            n = min(num_images, len(images))
            vis_images = images[:n]
            vis_masked = masked_img[:n]
            vis_recon = recon[:n]
            vis_composite = composite[:n]

        # Denormalize to [0, 1] for fair metrics
        orig_dn = _denormalize_tensor(images)
        recon_dn = _denormalize_tensor(recon)
        comp_dn = _denormalize_tensor(composite)

        # Full image metrics (composite vs original)
        all_mse_full.append(_mse(comp_dn, orig_dn))
        all_psnr_full.append(_psnr(comp_dn, orig_dn))
        all_ssim.append(_ssim_batch(comp_dn, orig_dn))

        # Masked-patches-only metrics: compare reconstructed patches where mask=1
        # Expand mask to pixel level
        mask_px = mask.unsqueeze(-1).float()  # [B, N, 1]
        orig_patches = _patchify(orig_dn)     # [B, N, P]
        recon_patches = _patchify(recon_dn)
        # Only compute over masked patches
        masked_orig = orig_patches[mask_px.expand_as(orig_patches) == 1]
        masked_recon = recon_patches[mask_px.expand_as(recon_patches) == 1]
        mse_m = F.mse_loss(masked_recon, masked_orig).item()
        psnr_m = 10 * np.log10(1.0 / mse_m) if mse_m > 0 else float("inf")
        all_mse_masked.append(mse_m)
        all_psnr_masked.append(psnr_m)

    metrics = {
        "mse_full": round(float(np.mean(all_mse_full)), 6),
        "mse_masked": round(float(np.mean(all_mse_masked)), 6),
        "psnr_full": round(float(np.mean(all_psnr_full)), 2),
        "psnr_masked": round(float(np.mean(all_psnr_masked)), 2),
        "ssim": round(float(np.mean(all_ssim)), 4),
    }

    print(f"  Metrics (avg over {eval_batches} batches):")
    print(f"    MSE (full):        {metrics['mse_full']:.6f}")
    print(f"    MSE (masked only): {metrics['mse_masked']:.6f}")
    print(f"    PSNR (full):       {metrics['psnr_full']:.2f} dB")
    print(f"    PSNR (masked):     {metrics['psnr_masked']:.2f} dB")
    print(f"    SSIM:              {metrics['ssim']:.4f}")

    # Visualization
    n = vis_images.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    col_titles = ["Original", f"Masked ({mask_ratio:.0%})", "Reconstructed", "Composite"]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        axes[i, 0].imshow(_denormalize(vis_images[i]))
        axes[i, 1].imshow(_denormalize(vis_masked[i]))
        axes[i, 2].imshow(_denormalize(vis_recon[i]))
        axes[i, 3].imshow(_denormalize(vis_composite[i]))
        for j in range(4):
            axes[i, j].axis("off")

    metrics_str = (f"MSE={metrics['mse_masked']:.4f}  "
                   f"PSNR={metrics['psnr_masked']:.1f}dB  "
                   f"SSIM={metrics['ssim']:.3f}")
    fig.suptitle(f"MAE Reconstruction — {dataset_name} (mask={mask_ratio:.0%})\n{metrics_str}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = out_dir / f"reconstruction_mae_{dataset_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {"encoder": "MAE", "plot": str(save_path), "mask_ratio": mask_ratio, **metrics}


# ---------------------------------------------------------------------------
# General nearest-neighbor retrieval (works for any encoder)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(encoder, loader, max_samples=None, layer=None):
    """Extract features, images, and labels."""
    all_features, all_images, all_labels = [], [], []
    collected = 0
    desc = f"Extracting ({encoder.name})"
    for images, labels in tqdm(loader, desc=desc, leave=False):
        if layer:
            features = encoder.extract_features_from_layer(images, layer)
        else:
            features = encoder.extract_features(images)
        all_features.append(features.cpu())
        all_images.append(images)
        all_labels.append(labels)
        collected += len(labels)
        if max_samples and collected >= max_samples:
            break
    return (torch.cat(all_features)[:max_samples],
            torch.cat(all_images)[:max_samples],
            torch.cat(all_labels)[:max_samples])


def _denormalize_auto(img: torch.Tensor) -> np.ndarray:
    """Best-effort denormalize: shift to [0,1] range."""
    img = img.cpu().clone()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def retrieval_reconstruction_evaluate(
    encoder: BaseEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    dataset_name: str,
    out_dir: Path,
    num_queries: int = 8,
    top_k: int = 5,
    layer: str | None = None,
) -> dict:
    """Nearest-neighbor retrieval with metrics.

    Metrics:
      - Precision@k: fraction of top-k neighbors sharing the query's class
      - Mean cosine similarity of top-k neighbors
    """
    print("  Extracting train features...")
    train_feats, train_imgs, train_labels = _extract_features(
        encoder, train_loader, max_samples=5000, layer=layer)
    print("  Extracting test features...")
    test_feats, test_imgs, test_labels = _extract_features(
        encoder, test_loader, max_samples=max(num_queries, 200), layer=layer)

    # Normalize for cosine similarity
    train_feats_n = F.normalize(train_feats, dim=1)
    test_feats_n = F.normalize(test_feats, dim=1)

    # Full similarity matrix for metrics
    sim = test_feats_n @ train_feats_n.T  # [N_test, N_train]
    topk_sim, topk_idx = sim.topk(top_k, dim=1)

    # Precision@k: does the neighbor share the same class?
    topk_labels = train_labels[topk_idx]  # [N_test, top_k]
    matches = (topk_labels == test_labels.unsqueeze(1)).float()
    precision_at_k = matches.mean().item()

    # Per-k precision
    precision_per_k = {}
    for k in range(1, top_k + 1):
        precision_per_k[f"P@{k}"] = matches[:, :k].mean().item()

    # Mean cosine similarity
    mean_sim = topk_sim.mean().item()

    metrics = {
        "precision_at_k": round(precision_at_k, 4),
        "mean_cosine_sim": round(mean_sim, 4),
        **{k: round(v, 4) for k, v in precision_per_k.items()},
    }

    print(f"  Metrics (over {len(test_feats)} queries, top-{top_k}):")
    for k, v in precision_per_k.items():
        print(f"    {k}: {v:.4f}")
    print(f"    Mean cosine sim: {mean_sim:.4f}")

    # Visualization (first num_queries)
    n = min(num_queries, len(test_imgs))
    cols = 1 + top_k
    fig, axes = plt.subplots(n, cols, figsize=(2.5 * cols, 2.5 * n))

    axes[0, 0].set_title("Query", fontsize=11, fontweight="bold")
    for k in range(top_k):
        axes[0, k + 1].set_title(f"NN-{k+1}", fontsize=11, fontweight="bold")

    for i in range(n):
        axes[i, 0].imshow(_denormalize_auto(test_imgs[i]))
        axes[i, 0].axis("off")
        for k in range(top_k):
            idx = topk_idx[i, k].item()
            is_match = matches[i, k].item() > 0
            axes[i, k + 1].imshow(_denormalize_auto(train_imgs[idx]))
            # Green border for class match, red for mismatch
            color = "#2ecc71" if is_match else "#e74c3c"
            for spine in axes[i, k + 1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            axes[i, k + 1].set_xticks([])
            axes[i, k + 1].set_yticks([])

    layer_tag = f" @ {layer}" if layer else ""
    enc_tag = encoder.name.lower().replace("-", "_").replace(" ", "_")
    p1 = precision_per_k.get("P@1", 0)
    fig.suptitle(
        f"NN Retrieval — {encoder.name}{layer_tag} — {dataset_name}\n"
        f"P@1={p1:.3f}  P@{top_k}={precision_at_k:.3f}  cos_sim={mean_sim:.3f}",
        fontsize=12, fontweight="bold")
    fig.tight_layout()

    layer_ftag = f"_{layer.replace('.', '_')}" if layer else ""
    save_path = out_dir / f"reconstruction_{dataset_name}{layer_ftag}_{enc_tag}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return {"encoder": encoder.name, "plot": str(save_path), "top_k": top_k, **metrics}
