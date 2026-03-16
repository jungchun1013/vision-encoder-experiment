"""Progressive fragment-completion segmentation via patch features.

Algorithm (mirrors fragment_v2 logic at the encoder patch level):
1. Extract ALL patch features from the original image
2. Identify "object patches" — those overlapping non-white pixels
3. At each level L, reveal P = 0.7^(8-L) of object patches (exponential schedule)
4. Run 2-means on only the revealed patches
5. Assign cluster labels, map back to full grid, compute IoU

Output: a plot of mean IoU vs fragmentation level (1–8).
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from models.registry import get_encoder
from wrappers.encoder import BaseEncoder


# --------------------------------------------------------------------------- #
# Ground-truth mask from original image
# --------------------------------------------------------------------------- #

def _get_foreground_mask(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """Binary mask: 1 = foreground (non-white), 0 = background."""
    bg = np.all(image >= threshold, axis=-1)  # [H, W]
    return (~bg).astype(np.float32)


# --------------------------------------------------------------------------- #
# Patch feature extraction — multi-encoder support
# --------------------------------------------------------------------------- #

_transform_cache: dict[str, object] = {}


def _get_patch_grid_size(encoder: BaseEncoder) -> tuple[int, int]:
    """Return (gh, gw) — the spatial grid of patches for the encoder."""
    model = encoder.model
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        return tuple(model.patch_embed.grid_size)
    if hasattr(model, "config") and hasattr(model.config, "image_size"):
        ps = model.config.patch_size
        img_sz = model.config.image_size
        g = img_sz // ps
        return (g, g)
    # CLIP visual
    if hasattr(model, "visual"):
        v = model.visual
        if hasattr(v, "conv1"):
            # infer from conv1 kernel
            ps = v.conv1.kernel_size[0]
            # need input size — assume 224
            g = 224 // ps
            return (g, g)
    return (14, 14)  # fallback


@torch.no_grad()
def _extract_patch_features(encoder: BaseEncoder, image_pil: Image.Image) -> torch.Tensor:
    """Extract patch-level features from a single image.

    Returns [N_patches, D] tensor (excluding CLS/prefix tokens).
    """
    enc_name = encoder.name
    if enc_name not in _transform_cache:
        _transform_cache[enc_name] = encoder.get_transform()
    transform = _transform_cache[enc_name]
    img_t = transform(image_pil).unsqueeze(0).to(encoder.device)

    model = encoder.model
    name = encoder.name

    # HuggingFace MAE
    if name == "MAE":
        output = model(pixel_values=img_t)
        return output.last_hidden_state[:, 1:][0].cpu()

    # DINO-v1 (torch.hub)
    if hasattr(model, "get_intermediate_layers"):
        outputs = model.get_intermediate_layers(img_t, n=1)
        return outputs[0][0, 1:].cpu()

    # OpenCLIP CLIP
    if name == "CLIP":
        visual = model.visual
        if hasattr(visual, "trunk") and hasattr(visual.trunk, "forward_features"):
            features = visual.trunk.forward_features(img_t)
            n_prefix = getattr(visual.trunk, "num_prefix_tokens", 1)
            return features[0, n_prefix:].cpu()
        if hasattr(visual, "transformer"):
            x = visual.conv1(img_t)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls_tok = visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tok, x], dim=1)
            x = x + visual.positional_embedding.unsqueeze(0)
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)
            return x[0, 1:].cpu()
        raise RuntimeError("Could not extract CLIP patch features")

    # timm-style ViT (DINOv2, SigLIP, ViT-sup, etc.)
    if hasattr(model, "forward_features"):
        features = model.forward_features(img_t)
        if hasattr(model, "num_prefix_tokens"):
            n_prefix = model.num_prefix_tokens
        elif hasattr(model, "cls_token") and model.cls_token is not None:
            n_prefix = 1
        else:
            n_prefix = 0
        return features[0, n_prefix:].cpu()

    raise RuntimeError(f"Patch feature extraction not supported for {encoder.name}")


# --------------------------------------------------------------------------- #
# Identify object patches in the encoder's patch grid
# --------------------------------------------------------------------------- #

def _get_object_patch_indices(
    image: np.ndarray, gh: int, gw: int, threshold: int = 250,
) -> list[int]:
    """Return flat indices of encoder patches that overlap with non-white pixels.

    Maps the original image (H, W) onto the encoder's (gh, gw) patch grid.
    A patch is an "object patch" if any pixel in its region is non-white.
    """
    H, W = image.shape[:2]
    patch_h = H / gh
    patch_w = W / gw

    object_indices = []
    for idx in range(gh * gw):
        i, j = divmod(idx, gw)
        r0, r1 = int(i * patch_h), int((i + 1) * patch_h)
        c0, c1 = int(j * patch_w), int((j + 1) * patch_w)
        region = image[r0:r1, c0:c1]
        if np.any(region < threshold):
            object_indices.append(idx)

    return object_indices


# --------------------------------------------------------------------------- #
# Progressive masking at patch level + 2-means + IoU
# --------------------------------------------------------------------------- #

def _progressive_segment_iou(
    patch_feats: torch.Tensor,
    gh: int, gw: int,
    object_indices: list[int],
    gt_mask: np.ndarray,
    seed: int = 42,
) -> list[float]:
    """Run progressive masking on patch features and compute IoU at each level.

    At each level L (1-8):
      - Reveal P = 0.7^(8-L) fraction of object patches
      - 2-means on revealed patches only
      - Map labels back to full image, compute IoU

    Returns list of 8 IoU values.
    """
    H, W = gt_mask.shape
    N = gh * gw
    feats_np = patch_feats.float().numpy()

    # Shuffle object indices with fixed seed
    rng = random.Random(seed)
    shuffled_obj = list(object_indices)
    rng.shuffle(shuffled_obj)

    ious = []
    for L in range(1, 9):
        P = 0.7 ** (8.0 - L)
        num_reveal = max(1, int(P * len(shuffled_obj)))
        revealed = shuffled_obj[:num_reveal]

        if len(revealed) < 2:
            # Not enough patches for 2-means
            ious.append(0.0)
            continue

        # Extract features of revealed patches
        revealed_feats = feats_np[revealed]

        # 2-means clustering on revealed patches
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        revealed_labels = kmeans.fit_predict(revealed_feats)  # [num_reveal]

        # Build full label grid: -1 = background (unrevealed/white)
        label_flat = np.full(N, -1, dtype=np.int32)
        for idx, lab in zip(revealed, revealed_labels):
            label_flat[idx] = lab
        label_grid = label_flat.reshape(gh, gw)

        # Upsample to original resolution
        # Use float for interpolation: -1 stays as bg
        label_img = np.array(
            Image.fromarray(label_grid.astype(np.int32).astype(np.int16))
                 .resize((W, H), Image.NEAREST)
        ).astype(np.int32)

        # Try both cluster assignments for foreground
        iou_best = 0.0
        for fg_label in [0, 1]:
            pred_fg = (label_img == fg_label).astype(np.float32)
            intersection = (pred_fg * gt_mask).sum()
            union = ((pred_fg + gt_mask) > 0).sum()
            iou = intersection / (union + 1e-8)
            iou_best = max(iou_best, iou)

        ious.append(float(iou_best))

    return ious


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def fragment_segmentation_evaluate(
    encoder_names: list[str],
    out_dir: Path,
    device: str = "cuda",
    data_root: str | None = None,
    seed: int = 42,
) -> dict:
    """Evaluate progressive fragment segmentation across encoders.

    For each image: extract patch features once, then progressively reveal
    object patches following P = 0.7^(8-L), run 2-means, compute IoU.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    frag_dir = Path(data_root) / "fragment_v2" if data_root else \
        Path(__file__).resolve().parent.parent / "data" / "fragment_v2"

    sample_dirs = sorted(
        p for p in frag_dir.iterdir()
        if p.is_dir() and (p / "original.png").exists()
    )
    print(f"  Found {len(sample_dirs)} fragment_v2 samples")

    # Load original images and ground-truth masks
    print("  Loading images...")
    images: list[tuple[np.ndarray, np.ndarray]] = []
    for sample_dir in sample_dirs:
        original = np.array(Image.open(sample_dir / "original.png").convert("RGB"))
        gt_mask = _get_foreground_mask(original)
        images.append((original, gt_mask))

    # Per-encoder, per-level IoU accumulation
    all_ious: dict[str, dict[int, list[float]]] = {}
    enc_display_names: dict[str, str] = {}

    # Process one encoder at a time
    for enc_name in encoder_names:
        try:
            encoder = get_encoder(enc_name, device=device)
            _ = encoder.model
            print(f"\n  === {encoder.name} ===")
        except Exception as e:
            print(f"  [SKIP] {enc_name}: {e}")
            continue

        enc_display_names[enc_name] = encoder.name
        all_ious[enc_name] = {L: [] for L in range(1, 9)}

        gh, gw = _get_patch_grid_size(encoder)
        print(f"  Patch grid: {gh}x{gw} = {gh * gw} patches")

        for si, (original, gt_mask) in enumerate(images):
            # Extract features from original image (once per image)
            original_pil = Image.fromarray(original)
            patch_feats = _extract_patch_features(encoder, original_pil)

            # Actual grid size from feature count
            N = patch_feats.shape[0]
            actual_gh, actual_gw = gh, gw
            if actual_gh * actual_gw != N:
                # Recompute from N
                actual_gw = int(round(N ** 0.5))
                actual_gh = actual_gw
                if actual_gh * actual_gw != N:
                    for h in range(int(N ** 0.5) + 1, 0, -1):
                        if N % h == 0:
                            actual_gh, actual_gw = h, N // h
                            break

            # Identify object patches
            object_indices = _get_object_patch_indices(
                original, actual_gh, actual_gw
            )

            # Progressive masking → IoU at each level
            level_ious = _progressive_segment_iou(
                patch_feats, actual_gh, actual_gw,
                object_indices, gt_mask, seed=seed,
            )

            for L_idx, iou in enumerate(level_ious):
                all_ious[enc_name][L_idx + 1].append(iou)

            if (si + 1) % 20 == 0 or si == 0 or si == len(images) - 1:
                mean_ious = [np.mean(all_ious[enc_name][L]) for L in range(1, 9)]
                ious_str = " ".join(f"{v:.3f}" for v in mean_ious)
                print(f"  [{si + 1}/{len(images)}] IoU = [{ious_str}]")

        # Free VRAM
        del encoder
        _transform_cache.clear()
        torch.cuda.empty_cache()

    if not all_ious:
        print("  No encoders available.")
        return {"plots": []}

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    levels = list(range(1, 9))

    for enc_name in all_ious:
        mean_ious = [np.mean(all_ious[enc_name][L]) for L in levels]
        std_ious = [np.std(all_ious[enc_name][L]) for L in levels]
        label = enc_display_names[enc_name]
        ax.plot(levels, mean_ious, "o-", label=label, linewidth=2, markersize=6)
        ax.fill_between(
            levels,
            [m - s for m, s in zip(mean_ious, std_ious)],
            [m + s for m, s in zip(mean_ious, std_ious)],
            alpha=0.15,
        )

    ax.set_xlabel("Fragmentation Level", fontsize=13)
    ax.set_ylabel("IoU (foreground)", fontsize=13)
    ax.set_title("Progressive Fragment Completion — 2-Means Segmentation IoU", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"L{L}\n({0.7**(8-L):.0%})" for L in levels], fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = out_dir / "fragment_segmentation_iou.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved plot: {plot_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Fragment Segmentation IoU Summary")
    print(f"{'='*70}")
    header = f"  {'Encoder':<15}" + "".join(f"{'L' + str(L):>8}" for L in levels)
    print(header)
    print(f"  {'-'*15}" + "-" * (8 * len(levels)))
    for enc_name in all_ious:
        row = f"  {enc_display_names[enc_name]:<15}"
        for L in levels:
            row += f"{np.mean(all_ious[enc_name][L]):>8.4f}"
        print(row)
    print()

    return {
        "plots": [str(plot_path)],
        "ious": {enc_name: {L: float(np.mean(all_ious[enc_name][L])) for L in levels}
                 for enc_name in all_ious},
    }
