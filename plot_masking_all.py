"""Plot all masking results on a single figure."""

import matplotlib.pyplot as plt
from pathlib import Path

# Data from the masking experiment runs
results = {
    "DINOv2":          [0.9881, 0.9779, 0.9641, 0.9466, 0.9214, 0.8881, 0.8534, 0.7972, 0.7229, 0.6025],
    "ViT-supervised":  [0.9732, 0.9724, 0.9671, 0.9595, 0.9459, 0.9266, 0.8970, 0.8515, 0.7838, 0.6385],
    "DINO-v1":         [0.9567, 0.9410, 0.9197, 0.8955, 0.8625, 0.8164, 0.7722, 0.7096, 0.6401, 0.5360],
    "SigLIP":          [0.9513, 0.9358, 0.9125, 0.8920, 0.8448, 0.7877, 0.7153, 0.6198, 0.5335, 0.4188],
    "CLIP":            [0.9291, 0.8996, 0.8504, 0.7817, 0.6977, 0.6037, 0.5424, 0.4805, 0.4444, 0.3901],
    "ResNet-50":       [0.8919, 0.8675, 0.8308, 0.7921, 0.7378, 0.6785, 0.6287, 0.5811, 0.5248, 0.4500],
    "MAE":             [0.7052, 0.6783, 0.6552, 0.6335, 0.6076, 0.5904, 0.5706, 0.5457, 0.5031, 0.4445],
}

mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Style: distinct colors and markers per encoder
styles = {
    "DINOv2":         {"color": "#1f77b4", "marker": "o"},
    "ViT-supervised": {"color": "#ff7f0e", "marker": "s"},
    "DINO-v1":        {"color": "#2ca02c", "marker": "^"},
    "SigLIP":         {"color": "#d62728", "marker": "D"},
    "CLIP":           {"color": "#9467bd", "marker": "v"},
    "ResNet-50":      {"color": "#8c564b", "marker": "P"},
    "MAE":            {"color": "#e377c2", "marker": "X"},
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, accs in results.items():
    s = styles[name]
    ax.plot(mask_ratios, accs, marker=s["marker"], color=s["color"],
            linewidth=2, markersize=7, label=name)

ax.set_xlabel("Mask Ratio", fontsize=13)
ax.set_ylabel("k-NN Accuracy", fontsize=13)
ax.set_title("Progressive Patch Masking — CIFAR-10 (all encoders)", fontsize=14)
ax.set_xticks(mask_ratios)
ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=10, loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_dir = Path(__file__).resolve().parent / "output" / "masking"
out_dir.mkdir(parents=True, exist_ok=True)
save_path = out_dir / "masking_cifar10_all.png"
fig.savefig(save_path, dpi=150)
print(f"Saved: {save_path}")
