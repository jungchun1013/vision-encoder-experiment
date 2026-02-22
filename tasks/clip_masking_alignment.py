"""CLIP image-text alignment under progressive masking.

Workflow (per image)
--------------------
1. Pre-encode all dataset images with CLIP (no normalisation)
2. Classify the full image → predicted class name (softmax)
3. Progressively mask patches → re-encode at each mask ratio
4. At each mask ratio, find the dataset image that best matches CLIP's
   prediction → "reference point" (what CLIP thinks the image is)
5. Normalise all embeddings, then t-SNE project:
   relevant text anchors ★ + reference images ■ + masked-image trajectory ●
6. Plot per-image figure (one image at a time, repeated for N images)

Works for ANY dataset:
  - Datasets with predefined classes (cifar10, stl10): uses those as vocab
  - Datasets without classes (fragment, etc.): CLIP auto-labels from a broad
    fallback vocabulary
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import open_clip

from datasets.loader import get_dataset
from tasks.attention_masking import _apply_patch_mask


# ---- vocabularies -------------------------------------------------------- #

_PREDEFINED_VOCAB: dict[str, list[str]] = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "stl10": [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck",
    ],
}

# Broad fallback vocabulary for datasets without predefined class names
_FALLBACK_VOCAB: list[str] = [
    # instruments
    "accordion", "guitar", "piano", "violin", "drum", "trumpet", "harp",
    "flute", "saxophone", "banjo", "harmonica", "organ", "cello", "tuba",
    # animals
    "bird", "cat", "dog", "horse", "deer", "frog", "monkey", "fish",
    "bear", "lion", "tiger", "elephant", "rabbit", "snake", "turtle",
    "butterfly", "spider", "bee", "dolphin", "whale", "penguin", "owl",
    "parrot", "fox", "wolf", "mouse", "squirrel", "bat", "crab", "lobster",
    # vehicles
    "airplane", "car", "truck", "ship", "boat", "bicycle", "motorcycle",
    "bus", "train", "helicopter", "submarine", "tractor", "ambulance",
    # household objects
    "clock", "watch", "lamp", "chair", "table", "bed", "sofa", "mirror",
    "cup", "bottle", "bowl", "plate", "fork", "knife", "spoon", "kettle",
    "ashtray", "vase", "basket", "box", "jar", "bucket", "broom", "candle",
    # tools & electronics
    "hammer", "screwdriver", "wrench", "scissors", "saw", "axe",
    "phone", "computer", "keyboard", "camera", "television", "radio",
    "microphone", "speaker", "headphones", "binoculars", "telescope",
    # food
    "apple", "banana", "orange", "grape", "strawberry", "pizza", "cake",
    "bread", "cheese", "ice cream", "watermelon", "pineapple", "cherry",
    # nature
    "flower", "tree", "mushroom", "leaf", "cactus", "palm tree",
    "mountain", "volcano", "waterfall", "rainbow", "cloud", "sun", "moon",
    # buildings & structures
    "house", "building", "bridge", "tower", "castle", "church", "windmill",
    "lighthouse", "tent", "pyramid", "fence", "gate",
    # clothing & accessories
    "hat", "shoe", "boot", "glove", "glasses", "crown", "ring", "necklace",
    "umbrella", "bag", "suitcase", "belt", "tie",
    # sports & toys
    "ball", "balloon", "kite", "teddy bear", "doll", "dice", "chess piece",
    # stationery
    "book", "pen", "pencil", "paintbrush", "envelope", "globe", "map",
    # weapons & armour
    "sword", "shield", "bow", "arrow", "cannon", "anchor",
    # misc
    "key", "lock", "bell", "trophy", "medal", "hourglass", "compass",
    "magnet", "rocket", "robot", "skull", "flag", "wheel", "gear",
    "stethoscope", "syringe", "pill", "fire extinguisher",
    "barrel", "coffin", "cross", "diamond", "feather", "hand",
    "heart", "lightning bolt", "magnifying glass", "scroll", "torch",
]


def _get_vocab(dataset_name: str) -> tuple[list[str], bool]:
    """Return (vocab, has_gt).

    has_gt is True when the vocab doubles as ground-truth class labels
    (i.e. ``dataset[i][1]`` is a valid index into vocab).
    """
    key = dataset_name.lower().replace("-", "").replace("_", "")
    if key in _PREDEFINED_VOCAB:
        return _PREDEFINED_VOCAB[key], True
    return _FALLBACK_VOCAB, False


# ---- t-SNE --------------------------------------------------------------- #

def _tsne_2d(embeddings: torch.Tensor, perplexity: float = 15.0) -> np.ndarray:
    """Project [N, D] → [N, 2] via t-SNE."""
    from sklearn.manifold import TSNE
    X = embeddings.float().numpy()
    perp = min(perplexity, max(2.0, len(X) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                init="pca", learning_rate="auto")
    return tsne.fit_transform(X)


# ---- helpers -------------------------------------------------------------- #

def _denormalize_display(img: torch.Tensor) -> np.ndarray:
    img = img.cpu().float()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def _encode_all_images(model, dataset, device: str, batch_size: int = 32) -> torch.Tensor:
    """Encode all images in a dataset → [N, D] (unnormalised)."""
    all_feats = []
    for i in range(0, len(dataset), batch_size):
        end = min(i + batch_size, len(dataset))
        batch = torch.stack([dataset[j][0] for j in range(i, end)])
        feats = model.encode_image(batch.to(device))
        all_feats.append(feats.cpu())
    return torch.cat(all_feats)  # [N, D]


# ---- main entry ---------------------------------------------------------- #

@torch.no_grad()
def clip_masking_alignment(
    dataset_name: str,
    out_dir: Path,
    device: str = "cuda",
    mask_ratios: list[float] | None = None,
    num_images: int = 8,
    data_root: str | None = None,
    seed: int = 42,
) -> dict:
    """Run the CLIP masking-alignment experiment (one plot per image).

    Normalisation strategy (same as original identify_object code):
      - encode_image / encode_text: no normalisation
      - classification: softmax on raw dot-product logits
      - t-SNE: F.normalize before projection
    """
    if mask_ratios is None:
        mask_ratios = [round(r * 0.1, 1) for r in range(10)]
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab, has_gt = _get_vocab(dataset_name)

    # ---- load CLIP -------------------------------------------------------- #
    print("  Loading CLIP ViT-L-14-336 …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-336", pretrained="openai",
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14-336")

    # ---- encode full vocabulary (no normalisation) ------------------------ #
    prompts = [f"{c}" for c in vocab]
    text_tokens = tokenizer(prompts).to(device)
    text_feats = model.encode_text(text_tokens).cpu()  # [V, D]
    print(f"  Vocab size: {len(vocab)}  (ground-truth labels: {has_gt})")

    # ---- load dataset & pre-encode all images ----------------------------- #
    dataset = get_dataset(
        dataset_name, train=False, transform=preprocess, data_root=data_root,
    )
    print("  Pre-encoding all dataset images …")
    all_full_feats = _encode_all_images(model, dataset, device)
    print(f"  Encoded {len(all_full_feats)} images")

    # ---- select images ---------------------------------------------------- #
    if has_gt:
        selected: list[tuple[int, int]] = []
        seen: set[int] = set()
        for i in range(len(dataset)):
            _, lbl = dataset[i]
            lbl = int(lbl)
            if lbl not in seen:
                selected.append((i, lbl))
                seen.add(lbl)
            if len(selected) >= num_images:
                break
    else:
        selected = [(i, int(dataset[i][1])) for i in range(min(num_images, len(dataset)))]

    print(f"  Selected {len(selected)} images")

    # ---- per-image loop --------------------------------------------------- #
    patch_size = 16
    plots: list[str] = []
    all_text_sims: dict[str, list[float]] = {}
    all_self_sims: dict[str, list[float]] = {}

    for img_i, (ds_idx, gt_label) in enumerate(selected):
        img, _ = dataset[ds_idx]

        # classify full (unmasked) image — softmax on raw logits
        full_feat = all_full_feats[ds_idx]  # [D], unnormalised
        logits_full = full_feat @ text_feats.T  # [V]
        probs_full = logits_full.softmax(dim=-1)
        pred_idx = probs_full.argmax().item()
        pred_name = vocab[pred_idx]
        print(f"  Full image: \"{pred_name}\" (P={probs_full[pred_idx].item():.3f})")

        anchor_idx = gt_label if has_gt else pred_idx
        anchor_name = vocab[anchor_idx] if has_gt else pred_name
        anchor_feat = text_feats[anchor_idx]

        # encode at every mask ratio + find references
        feats_per_ratio: list[torch.Tensor] = []
        text_cos_sims: list[float] = []
        self_cos_sims: list[float] = []
        pred_per_ratio: list[int] = []
        ref_img_indices: list[int] = []

        for ratio in mask_ratios:
            masked, _ = _apply_patch_mask(img, ratio, patch_size, seed=seed + ds_idx)
            feat = model.encode_image(masked.unsqueeze(0).to(device))
            feat = feat.cpu().squeeze(0)  # [D], unnormalised
            feats_per_ratio.append(feat)
            text_cos_sims.append((feat @ anchor_feat).item())
            self_cos_sims.append((feat @ full_feat).item())

            # softmax classification of masked image
            logits = feat @ text_feats.T
            probs = logits.softmax(dim=-1)
            masked_pred_idx = probs.argmax().item()
            pred_per_ratio.append(masked_pred_idx)

            # find reference: dataset image best matching predicted text
            ref_sims = text_feats[masked_pred_idx] @ all_full_feats.T
            ref_img_indices.append(ref_sims.argmax().item())

        # log
        if has_gt:
            gt_name = vocab[gt_label]
            tag = "✓" if pred_name == gt_name else f"✗→{pred_name}"
            label_str = gt_name
            print(f"    [{gt_name}] {tag}  text@0%={text_cos_sims[0]:.3f}  text@90%={text_cos_sims[-1]:.3f}  self@0%={self_cos_sims[0]:.3f}  self@90%={self_cos_sims[-1]:.3f}")
        else:
            label_str = f"img{ds_idx:03d}"
            print(f"    [img {ds_idx:03d}] → \"{pred_name}\"  text@0%={text_cos_sims[0]:.3f}  text@90%={text_cos_sims[-1]:.3f}  self@0%={self_cos_sims[0]:.3f}  self@90%={self_cos_sims[-1]:.3f}")

        all_text_sims[label_str] = text_cos_sims
        all_self_sims[label_str] = self_cos_sims

        # ---- collect points for t-SNE ------------------------------------- #
        # text anchors: top-K similar + all per-ratio predictions
        topk_text = logits_full.topk(k=min(10, len(vocab))).indices.tolist()
        text_to_show = set(topk_text) | set(pred_per_ratio)
        if has_gt:
            text_to_show.add(gt_label)
        text_to_show = sorted(text_to_show)
        n_txt = len(text_to_show)
        txt_idx_map = {vi: i for i, vi in enumerate(text_to_show)}
        text_feats_local = text_feats[text_to_show]  # [n_txt, D]

        # unique reference images
        unique_refs = sorted(set(ref_img_indices))
        ref_local_map = {ri: i for i, ri in enumerate(unique_refs)}
        n_ref = len(unique_refs)
        ref_feats_local = all_full_feats[unique_refs]  # [n_ref, D]

        # label each reference by the prediction that selected it
        ref_labels: dict[int, str] = {}
        for ri_ratio, ri_ref in enumerate(ref_img_indices):
            ref_labels.setdefault(ri_ref, vocab[pred_per_ratio[ri_ratio]])

        print(f"      t-SNE: {n_txt} text + {n_ref} ref + {len(mask_ratios)} traj")

        # t-SNE — normalise all embeddings before projection
        tsne_input = torch.cat([text_feats_local, ref_feats_local,
                                torch.stack(feats_per_ratio)])
        tsne_input = tsne_input / tsne_input.norm(dim=-1, keepdim=True)
        proj = _tsne_2d(tsne_input)

        text_proj = proj[:n_txt]
        ref_proj = proj[n_txt:n_txt + n_ref]
        traj_proj = proj[n_txt + n_ref:]

        # ---- PLOT --------------------------------------------------------- #
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        ratio_cmap = plt.cm.RdYlGn_r

        # -- Left: t-SNE scatter -------------------------------------------- #
        # text anchors (stars)
        for ti, vocab_idx in enumerate(text_to_show):
            is_anchor = (vocab_idx == anchor_idx)
            ax1.scatter(
                text_proj[ti, 0], text_proj[ti, 1],
                marker="*", s=350 if is_anchor else 120,
                c="gold" if is_anchor else "lightgray",
                edgecolors="black", linewidths=0.8, zorder=10,
                alpha=1.0 if is_anchor else 0.5,
            )
            ax1.annotate(
                f'"{vocab[vocab_idx]}"',
                (text_proj[ti, 0], text_proj[ti, 1]),
                fontsize=7 if is_anchor else 6,
                fontweight="bold" if is_anchor else "normal",
                color="black" if is_anchor else "gray",
                xytext=(6, 6), textcoords="offset points",
            )

        # reference images (squares)
        for ref_idx, pos_i in ref_local_map.items():
            rp = ref_proj[pos_i]
            ax1.scatter(rp[0], rp[1], marker="s", s=100, c="lightskyblue",
                        edgecolors="black", linewidths=0.8, zorder=9)
            ax1.annotate(
                f"[{ref_labels.get(ref_idx, '?')}]",
                (rp[0], rp[1]),
                fontsize=6, fontstyle="italic", color="steelblue",
                xytext=(5, -8), textcoords="offset points",
            )

        # dotted lines: trajectory → reference at each ratio
        for ri in range(len(mask_ratios)):
            rp = ref_proj[ref_local_map[ref_img_indices[ri]]]
            ax1.plot([traj_proj[ri, 0], rp[0]], [traj_proj[ri, 1], rp[1]],
                     color="steelblue", alpha=0.15, linestyle=":",
                     linewidth=0.8, zorder=2)

        # image trajectory (circles)
        ax1.plot(traj_proj[:, 0], traj_proj[:, 1],
                 color="dimgray", alpha=0.4, linewidth=1.5, zorder=3)
        for ri, ratio in enumerate(mask_ratios):
            ax1.scatter(
                traj_proj[ri, 0], traj_proj[ri, 1],
                c=[ratio_cmap(ratio)], s=80 if ri == 0 else 30,
                edgecolors="black", linewidths=0.6,
                alpha=max(1.0 - 0.5 * ratio, 0.3), zorder=5,
            )
        ax1.annotate("0%", (traj_proj[0, 0], traj_proj[0, 1]),
                     fontsize=7, color="green", fontweight="bold",
                     xytext=(-12, -12), textcoords="offset points")
        ax1.annotate(f"{mask_ratios[-1]:.0%}",
                     (traj_proj[-1, 0], traj_proj[-1, 1]),
                     fontsize=7, color="red",
                     xytext=(4, -12), textcoords="offset points")

        ax1.set_title("CLIP Embedding Space (t-SNE)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("t-SNE 1", fontsize=10)
        ax1.set_ylabel("t-SNE 2", fontsize=10)
        ax1.grid(True, alpha=0.2)

        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markersize=14, markeredgecolor="black", label="Anchor text"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="lightgray",
                   markersize=10, markeredgecolor="black", label="Other text"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="lightskyblue",
                   markersize=10, markeredgecolor="black", label="Reference image"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
                   markersize=8, markeredgecolor="black", label="Image (0% mask)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=8, markeredgecolor="black",
                   label=f"Image ({mask_ratios[-1]:.0%} mask)"),
        ]
        ax1.legend(handles=legend, fontsize=8, loc="best")

        # -- Right: similarity curves --------------------------------------- #
        ax2.plot(mask_ratios, text_cos_sims, marker="o", markersize=4,
                 linewidth=2, color="steelblue", label="text sim")
        ax2.plot(mask_ratios, self_cos_sims, marker="s", markersize=4,
                 linewidth=2, color="coral", label="self sim")

        # annotate when prediction changes
        prev_pred = None
        for ri, ratio in enumerate(mask_ratios):
            pred = vocab[pred_per_ratio[ri]]
            if pred != prev_pred:
                ax2.annotate(
                    f'"{pred}"', (ratio, text_cos_sims[ri]),
                    fontsize=7, fontweight="bold", ha="center",
                    color="darkred",
                    xytext=(0, 14), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                )
                prev_pred = pred

        ax2.set_title("Similarity Under Masking", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Mask Ratio", fontsize=10)
        ax2.set_ylabel("Dot-Product Similarity", fontsize=10)
        ax2.set_xticks(mask_ratios)
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(True, alpha=0.3)

        # title
        if has_gt:
            title_label = f'{vocab[gt_label]} (pred: "{pred_name}")'
        else:
            title_label = f'img {ds_idx:03d} → "{pred_name}"'
        fig.suptitle(
            f"CLIP Progressive Masking — {dataset_name} — {title_label}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = out_dir / f"clip_alignment_{dataset_name}_{img_i:02d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(save_path))
        print(f"  Saved: {save_path}")

    return {
        "plots": plots,
        "text_cos_sims": all_text_sims,
        "self_cos_sims": all_self_sims,
    }


def _save_example_strip(
    img: torch.Tensor, mask_ratios: list[float], patch_size: int,
    seed: int, ds_idx: int, class_name: str, save_path: Path,
):
    """Save a horizontal strip showing one image at every mask ratio."""
    n = len(mask_ratios)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.5))
    for i, ratio in enumerate(mask_ratios):
        masked, _ = _apply_patch_mask(img, ratio, patch_size, seed=seed + ds_idx)
        axes[i].imshow(_denormalize_display(masked))
        axes[i].set_title(f"{ratio:.0%}", fontsize=10)
        axes[i].axis("off")
    fig.suptitle(f'"{class_name}" — progressive masking', fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
