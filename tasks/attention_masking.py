"""Progressive masking attention visualization.

For each ViT encoder, progressively mask image patches and visualize where
the CLS token's attention is directed (last transformer block, averaged over
heads). This reveals:

- How attention redistributes as information is removed
- Which encoders maintain focus on salient regions under heavy masking
- Differences in attention robustness across pre-training strategies

Output per sample image: a grid figure with
  - Top row: masked input images at each ratio
  - One row per encoder: CLS->patch attention heatmap overlaid on original
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from datasets.loader import get_dataset
from models.registry import get_encoder
from wrappers.encoder import BaseEncoder


# --------------------------------------------------------------------------- #
# Patch masking
# --------------------------------------------------------------------------- #

def _apply_patch_mask(
    image: torch.Tensor, mask_ratio: float, patch_size: int, seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-out random patches of an image tensor [C, H, W].

    Returns (masked_image, mask_grid [gh, gw]) where 1 = masked.
    """
    C, H, W = image.shape
    gh, gw = H // patch_size, W // patch_size
    n_patches = gh * gw

    if mask_ratio <= 0:
        return image, torch.zeros(gh, gw)

    n_mask = int(n_patches * mask_ratio)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_patches, generator=gen)

    mask = torch.zeros(n_patches)
    mask[perm[:n_mask]] = 1.0
    mask = mask.reshape(gh, gw)

    masked = image.clone()
    for i in range(gh):
        for j in range(gw):
            if mask[i, j] > 0:
                masked[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ] = 1.0
    return masked, mask


# --------------------------------------------------------------------------- #
# Attention extraction  (last transformer block, all heads)
# --------------------------------------------------------------------------- #

def _timm_attn_hook(module, input, output, storage: dict):
    """Forward hook for a timm ``Attention`` module.

    Re-computes q @ k^T / sqrt(d) from the module's ``qkv`` projection so
    that the softmax attention weights are captured even when the model uses
    ``F.scaled_dot_product_attention`` (which does not expose weights).
    """
    x = input[0]  # [B, N, C]
    B, N, C = x.shape
    head_dim = getattr(module, "head_dim", C // module.num_heads)
    qkv = module.qkv(x)  # [B, N, 3 * num_heads * head_dim]
    qkv = qkv.reshape(B, N, 3, module.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _v = qkv.unbind(0)
    # Apply QK normalization when present (newer timm models)
    if hasattr(module, "q_norm") and not isinstance(module.q_norm, torch.nn.Identity):
        q = module.q_norm(q)
    if hasattr(module, "k_norm") and not isinstance(module.k_norm, torch.nn.Identity):
        k = module.k_norm(k)
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)  # [B, heads, N, N]
    storage["attn"] = attn.cpu()


@torch.no_grad()
def _extract_attention_timm(model, images: torch.Tensor, forward_fn=None):
    """Capture attention weights from the last block of a timm-style ViT.

    Parameters
    ----------
    model : nn.Module
        Must have ``model.blocks`` (list of transformer blocks).
    images : torch.Tensor
        Input batch [B, C, H, W].
    forward_fn : callable, optional
        Custom forward (e.g. ``model.encode_image``).  If *None*, uses
        ``model(images)``.

    Returns
    -------
    torch.Tensor  [B, heads, N, N]
    """
    storage: dict = {}
    last_attn = model.blocks[-1].attn
    handle = last_attn.register_forward_hook(
        lambda m, i, o: _timm_attn_hook(m, i, o, storage)
    )
    try:
        (forward_fn or (lambda: model(images)))()
    finally:
        handle.remove()
    return storage["attn"]


@torch.no_grad()
def _extract_attention_clip(model, images: torch.Tensor):
    """Capture attention from an OpenCLIP CLIP model.

    Tries the timm-backed ``visual.trunk.blocks`` path first, then falls
    back to the original CLIP ``visual.transformer.resblocks`` path.
    """
    visual = model.visual

    # --- timm-backed open_clip (newer) ---
    if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        return _extract_attention_timm(
            visual.trunk, images,
            forward_fn=lambda: model.encode_image(images),
        )

    # --- original CLIP-style with nn.MultiheadAttention ---
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        last_block = visual.transformer.resblocks[-1]
        storage: dict = {}
        orig_fn = last_block.attention

        def _patched_attention(q_x, k_x=None, v_x=None, attn_mask=None):
            k_x = k_x if k_x is not None else q_x
            v_x = v_x if v_x is not None else q_x
            out, weights = last_block.attn(
                q_x, k_x, v_x,
                need_weights=True, average_attn_weights=False,
                attn_mask=attn_mask,
            )
            storage["attn"] = weights.cpu()  # [B, heads, N, N]
            return out

        last_block.attention = _patched_attention
        try:
            model.encode_image(images)
        finally:
            last_block.attention = orig_fn
        return storage["attn"]

    raise RuntimeError("Could not locate attention layers in the CLIP visual encoder")


@torch.no_grad()
def _extract_attention_mae(model, images: torch.Tensor):
    """Capture attention from a HuggingFace ViTMAE model."""
    output = model(pixel_values=images, output_attentions=True)
    return output.attentions[-1].cpu()  # [B, heads, N, N]


# --------------------------------------------------------------------------- #
# Unified extraction  ->  [B, grid_h, grid_w]
# --------------------------------------------------------------------------- #

_TIMM_ENCODERS = {"DINOv2", "ViT-supervised", "DINO-v1", "SigLIP"}


def _has_cls_token(encoder: BaseEncoder) -> bool:
    """Return True if the model prepends a CLS token to the patch sequence."""
    m = encoder.model
    if hasattr(m, "cls_token") and m.cls_token is not None:
        return True
    if hasattr(m, "num_prefix_tokens") and m.num_prefix_tokens > 0:
        return True
    # HuggingFace MAE always has CLS
    if encoder.name == "MAE":
        return True
    return False


def extract_cls_patch_attention(
    encoder: BaseEncoder, images: torch.Tensor,
) -> torch.Tensor:
    """Extract CLS -> patch attention map (or aggregate if no CLS).

    Returns
    -------
    torch.Tensor  [B, grid_h, grid_w]
        Per-patch attention intensity in [0, 1].
    """
    images = images.to(encoder.device)
    model = encoder.model
    name = encoder.name

    # 1) Get raw attention  [B, heads, N, N]
    if name in _TIMM_ENCODERS:
        attn = _extract_attention_timm(model, images)
    elif name == "CLIP":
        attn = _extract_attention_clip(model, images)
    elif name == "MAE":
        attn = _extract_attention_mae(model, images)
    else:
        raise ValueError(f"Attention extraction not implemented for '{name}'")

    # 2) Derive a per-patch intensity vector  [B, num_patches]
    has_cls = _has_cls_token(encoder)
    if has_cls:
        # CLS is token 0 → its attention to spatial tokens 1:
        cls_attn = attn[:, :, 0, 1:]          # [B, heads, num_patches]
        cls_attn = cls_attn.mean(dim=1)         # [B, num_patches]
    else:
        # No CLS: average "how much attention does each patch *receive*"
        cls_attn = attn.mean(dim=1).mean(dim=1)  # [B, N]

    # 3) Reshape to spatial grid
    B, N = cls_attn.shape
    gh = gw = int(round(N ** 0.5))
    if gh * gw != N:
        # Non-square grid (shouldn't normally happen for standard ViTs)
        gh = gw = int(N ** 0.5)
        cls_attn = cls_attn[:, : gh * gw]
    return cls_attn.reshape(B, gh, gw)


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #

def _denormalize(img: torch.Tensor) -> np.ndarray:
    """[C, H, W] tensor -> [H, W, C] numpy in [0, 1]."""
    img = img.cpu().float()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def _attention_overlay(
    image_np: np.ndarray,
    attn_map: np.ndarray,
    alpha: float = 0.55,
    pct_lo: float = 2.0,
    pct_hi: float = 98.0,
) -> np.ndarray:
    """Overlay an attention heatmap onto an RGB image.

    Uses **percentile** normalisation so that a single artifact token
    (common in DINOv2) does not wash out the rest of the heatmap.

    Parameters
    ----------
    image_np : [H, W, 3]  in [0, 1]
    attn_map : [gh, gw]  raw attention weights
    alpha : heatmap opacity
    pct_lo, pct_hi : percentile range for colour-scale clipping
    """
    H, W = image_np.shape[:2]
    # Bilinear upsample to image resolution
    attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8))
    attn_resized = np.asarray(attn_pil.resize((W, H), Image.BILINEAR)) / 255.0
    # Percentile normalisation — robust to outlier artifact tokens
    lo = np.percentile(attn_resized, pct_lo)
    hi = np.percentile(attn_resized, pct_hi)
    attn_resized = np.clip((attn_resized - lo) / (hi - lo + 1e-8), 0, 1)
    # Jet colourmap (matches DINO / DINOv2 paper conventions)
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    overlay = (1 - alpha) * image_np + alpha * heatmap
    return np.clip(overlay, 0, 1)


def _get_encoder_patch_size(encoder: BaseEncoder) -> int:
    """Return the spatial patch size (pixels) for the encoder's ViT."""
    m = encoder.model
    if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "proj"):
        return m.patch_embed.proj.kernel_size[0]
    # HuggingFace MAE
    if hasattr(m, "config") and hasattr(m.config, "patch_size"):
        return m.config.patch_size
    return 16


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def attention_masking_visualize(
    encoder_names: list[str],
    dataset_name: str,
    out_dir: Path,
    device: str = "cuda",
    mask_ratios: list[float] | None = None,
    image_indices: list[int] | None = None,
    data_root: str | None = None,
    seed: int = 42,
) -> dict:
    """Produce progressive-masking attention heatmap grids.

    Parameters
    ----------
    encoder_names : list[str]
        Registry keys (e.g. ``["dinov2", "clip", "mae"]``).  Non-ViT
        encoders are silently skipped.
    dataset_name, out_dir, device, data_root
        Standard testbed arguments.
    mask_ratios : list[float]
        Fraction of patches to zero-out at each step.
    image_indices : list[int]
        Which test-set images to visualise.
    seed : int
        RNG seed for reproducible masking patterns.

    Returns
    -------
    dict  with ``"plots"`` list of saved figure paths.
    """
    if mask_ratios is None:
        mask_ratios = [0.0, 0.25, 0.50, 0.75]
    if image_indices is None:
        image_indices = [0, 1, 2, 3]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip encoders without self-attention
    skip = {"resnet", "simclr"}
    vit_names = [n for n in encoder_names if n not in skip]

    # ---- Load encoders ----
    encoders: dict[str, BaseEncoder] = {}
    enc_transforms: dict[str, object] = {}
    for name in vit_names:
        try:
            enc = get_encoder(name, device=device)
            _ = enc.model  # trigger lazy load
            encoders[name] = enc
            enc_transforms[name] = enc.get_transform()
            print(f"  Loaded {enc.name}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if not encoders:
        print("  No ViT encoders available.")
        return {"plots": []}

    # ---- Display dataset (un-normalised, 224x224) for the top-row images ----
    from torchvision import transforms as T

    display_tf = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    display_ds = get_dataset(
        dataset_name, train=False, transform=display_tf, data_root=data_root,
    )

    # ---- Per-encoder datasets (with correct normalisation) ----
    enc_datasets: dict[str, object] = {}
    for name, tf in enc_transforms.items():
        enc_datasets[name] = get_dataset(
            dataset_name, train=False, transform=tf, data_root=data_root,
        )

    # ---- Generate figures ----
    n_ratios = len(mask_ratios)
    n_enc = len(encoders)
    save_paths: list[str] = []

    for img_idx in image_indices:
        display_img, label = display_ds[img_idx]
        display_np = _denormalize(display_img)

        fig, axes = plt.subplots(
            1 + n_enc, n_ratios,
            figsize=(3.2 * n_ratios, 3.0 * (1 + n_enc)),
        )
        if n_ratios == 1:
            axes = axes[:, None]

        # ---- Top row: masked inputs (using 16×16 patches for display) ----
        for col, ratio in enumerate(mask_ratios):
            masked_disp, _ = _apply_patch_mask(
                display_img, ratio, patch_size=16, seed=seed + img_idx,
            )
            axes[0, col].imshow(_denormalize(masked_disp))
            axes[0, col].set_title(f"mask {ratio:.0%}", fontsize=11, fontweight="bold")
            axes[0, col].axis("off")
        axes[0, 0].set_ylabel(
            "Input", fontsize=11, fontweight="bold",
            rotation=0, labelpad=50, va="center",
        )

        # ---- Encoder rows ----
        for row_off, (enc_name, encoder) in enumerate(encoders.items(), start=1):
            ps = _get_encoder_patch_size(encoder)
            enc_img, _ = enc_datasets[enc_name][img_idx]

            for col, ratio in enumerate(mask_ratios):
                masked_enc, _ = _apply_patch_mask(
                    enc_img, ratio, patch_size=ps, seed=seed + img_idx,
                )
                batch = masked_enc.unsqueeze(0)

                try:
                    attn_grid = extract_cls_patch_attention(encoder, batch)[0]
                    overlay = _attention_overlay(display_np, attn_grid.numpy())
                    axes[row_off, col].imshow(overlay)
                except Exception as e:
                    axes[row_off, col].text(
                        0.5, 0.5, f"err", color="red",
                        ha="center", va="center",
                        transform=axes[row_off, col].transAxes,
                    )
                    print(f"    [WARN] {encoder.name} mask={ratio}: {e}")

                axes[row_off, col].axis("off")

            axes[row_off, 0].set_ylabel(
                encoder.name, fontsize=11, fontweight="bold",
                rotation=0, labelpad=50, va="center",
            )

        fig.suptitle(
            f"CLS Attention Under Progressive Masking\n"
            f"{dataset_name}  —  image #{img_idx}  (label {label})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0.08, 0, 1, 0.94])

        path = out_dir / f"attention_masking_{dataset_name}_img{img_idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        save_paths.append(str(path))
        print(f"  Saved: {path}")

    return {"plots": save_paths}
