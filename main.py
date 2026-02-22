import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import models  # triggers all @register decorators
from models.registry import get_encoder, list_encoders
from datasets.loader import get_dataset, get_loaders, get_num_classes
from tasks.knn import knn_evaluate
from tasks.linear_probe import linear_probe_evaluate
from tasks.tsne import tsne_evaluate
from tasks.masking import masking_evaluate
from tasks.cka import cka_cross_encoder, cka_cross_layer
from tasks.reconstruction import mae_reconstruction_evaluate, retrieval_reconstruction_evaluate
from tasks.attention_masking import attention_masking_visualize
from tasks.clip_masking_alignment import clip_masking_alignment
from tasks.finetune import finetune_evaluate, _build_train_transform
from tasks.color_tsne import color_tsne_evaluate

TASKS = {
    "knn": "k-NN",
    "linear_probe": "Linear Probe",
    "tsne": "t-SNE",
    "masking": "Masking",
    "cka": "CKA",
    "reconstruction": "Reconstruction",
    "attention_masking": "Attention Masking",
    "clip_alignment": "CLIP Masking Alignment",
    "finetune": "Fine-tune",
    "color_tsne": "Color t-SNE",
}

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def _detect_block_layers(encoder) -> list[str]:
    """Auto-detect top-level block layers for an encoder."""
    all_layers = encoder.list_layers()
    layers = [l for l in all_layers
              if (l.startswith("blocks.") and l.count(".") == 1)                          # timm ViT
              or (l.startswith("encoder.layer.") and l.count(".") == 2)                    # HF MAE
              or (l.startswith("visual.transformer.resblocks.") and l.count(".") == 3)     # OpenCLIP
              or (l.startswith("layer") and l.count(".") == 0)]                            # ResNet
    if not layers:
        layers = [l for l in all_layers if l.count(".") == 0]
    return layers


def run_single(encoder_name: str, task: str, args, layer: str | None = None) -> dict:
    """Run a single encoder + task combination. Returns results dict."""
    encoder = get_encoder(encoder_name, device=args.device)
    layer_info = f"  |  Layer: {layer}" if layer else ""
    print(f"\n{'='*60}")
    print(f"Encoder: {encoder.name}  |  Task: {TASKS[task]}  |  Dataset: {args.dataset}{layer_info}")
    print(f"{'='*60}")

    print(f"Loading model...")
    transform = encoder.get_transform()

    print(f"Loading dataset...")
    num_classes = get_num_classes(args.dataset)
    start = time.time()

    if task == "knn":
        train_loader, test_loader = get_loaders(
            args.dataset, transform, batch_size=args.batch_size, data_root=args.data_root,
        )
        results = knn_evaluate(encoder, train_loader, test_loader,
                               num_classes=num_classes, layer=layer)

    elif task == "linear_probe":
        train_loader, test_loader = get_loaders(
            args.dataset, transform, batch_size=args.batch_size, data_root=args.data_root,
        )
        results = linear_probe_evaluate(
            encoder, train_loader, test_loader,
            num_classes=num_classes, epochs=args.epochs, layer=layer,
        )

    elif task == "finetune":
        train_transform = _build_train_transform(transform)
        train_loader = DataLoader(
            get_dataset(args.dataset, train=True, transform=train_transform, data_root=args.data_root),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
        )
        test_loader = DataLoader(
            get_dataset(args.dataset, train=False, transform=transform, data_root=args.data_root),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )
        results = finetune_evaluate(
            encoder, train_loader, test_loader,
            num_classes=num_classes, epochs=args.ft_epochs, lr=args.ft_lr,
        )

    elif task == "tsne":
        from torch.utils.data import DataLoader
        test_ds = get_dataset(args.dataset, train=False, transform=transform, data_root=args.data_root)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
        out_dir = OUTPUT_ROOT / "tsne"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = tsne_evaluate(
            encoder, test_loader, args.dataset, num_classes, out_dir,
            layer=layer, max_samples=args.max_samples, perplexity=args.perplexity,
        )

    elif task == "color_tsne":
        out_dir = OUTPUT_ROOT / "color_tsne"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = color_tsne_evaluate(encoder, out_dir, color_space=args.color_space,
                                      reduction=args.reduction,
                                      perplexity=args.perplexity)

    elif task == "masking":
        train_ds = get_dataset(args.dataset, train=True, transform=transform, data_root=args.data_root)
        test_ds = get_dataset(args.dataset, train=False, transform=transform, data_root=args.data_root)
        out_dir = OUTPUT_ROOT / "masking"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = masking_evaluate(
            encoder, train_ds, test_ds, args.dataset, num_classes, out_dir,
            batch_size=args.batch_size, patch_size=args.patch_size,
            layer=layer,
        )

    elif task == "reconstruction":
        out_dir = OUTPUT_ROOT / "reconstruction"
        out_dir.mkdir(parents=True, exist_ok=True)
        if encoder_name == "mae":
            # MAE: actual pixel reconstruction via encoder+decoder
            from torch.utils.data import DataLoader
            test_ds = get_dataset(args.dataset, train=False, transform=transform, data_root=args.data_root)
            test_loader = DataLoader(test_ds, batch_size=args.num_images, shuffle=True,
                                     num_workers=4, pin_memory=True)
            results = mae_reconstruction_evaluate(
                encoder, test_loader, args.dataset, out_dir,
                num_images=args.num_images, mask_ratio=args.mask_ratio,
            )
        else:
            # Other encoders: nearest-neighbor retrieval
            train_loader, test_loader = get_loaders(
                args.dataset, transform, batch_size=args.batch_size, data_root=args.data_root,
            )
            results = retrieval_reconstruction_evaluate(
                encoder, train_loader, test_loader, args.dataset, out_dir,
                num_queries=args.num_images, layer=layer,
            )

    elapsed = time.time() - start
    results["time_sec"] = round(elapsed, 1)
    results["encoder"] = encoder.name
    if layer:
        results["layer"] = layer
    return results


def print_results_table(all_results: list[dict], task: str):
    """Pretty-print a results table."""
    if "accuracy" not in all_results[0]:
        return
    has_layer = any(r.get("layer") for r in all_results)
    print(f"\n{'='*70}")
    print(f"Results: {TASKS[task]}")
    print(f"{'='*70}")
    if has_layer:
        print(f"{'Encoder':<20} {'Layer':<25} {'Accuracy':>10} {'Time (s)':>10}")
        print(f"{'-'*20} {'-'*25} {'-'*10} {'-'*10}")
        for r in sorted(all_results, key=lambda x: x.get("accuracy", 0), reverse=True):
            layer = r.get("layer", "-")
            print(f"{r['encoder']:<20} {layer:<25} {r['accuracy']:>10.4f} {r['time_sec']:>10.1f}")
    else:
        print(f"{'Encoder':<20} {'Accuracy':>10} {'Time (s)':>10}")
        print(f"{'-'*20} {'-'*10} {'-'*10}")
        for r in sorted(all_results, key=lambda x: x.get("accuracy", 0), reverse=True):
            print(f"{r['encoder']:<20} {r['accuracy']:>10.4f} {r['time_sec']:>10.1f}")
    print()


def _run_cka(encoder_names: list[str], args):
    """Run CKA analysis: cross-encoder or cross-layer."""
    from torch.utils.data import DataLoader

    out_dir = OUTPUT_ROOT / "cka"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    if args.layers:
        # Cross-layer mode: single encoder, multiple layers
        enc_name = encoder_names[0]
        encoder = get_encoder(enc_name, device=args.device)
        transform = encoder.get_transform()
        test_ds = get_dataset(args.dataset, train=False, transform=transform, data_root=args.data_root)
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        # --layers all: auto-detect block-level layers
        if "all" in args.layers:
            layers = _detect_block_layers(encoder)
            print(f"\nAuto-detected {len(layers)} layers: {layers}")
        else:
            layers = args.layers

        print(f"\nCross-layer CKA: {encoder.name} — {len(layers)} layers")
        result = cka_cross_layer(
            encoder, loader, layers, args.dataset, out_dir,
            max_samples=args.max_samples,
        )
    else:
        # Cross-encoder mode: multiple encoders
        if len(encoder_names) < 2:
            print("CKA cross-encoder mode requires at least 2 encoders.")
            print("  Use: --encoder dinov2 clip resnet")
            print("  Or for cross-layer: --encoder dinov2 --layers blocks.0 blocks.6 blocks.11")
            return

        encoders = []
        loaders = {}
        for name in encoder_names:
            try:
                enc = get_encoder(name, device=args.device)
                transform = enc.get_transform()
                test_ds = get_dataset(args.dataset, train=False, transform=transform, data_root=args.data_root)
                loaders[enc.name] = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                               num_workers=4, pin_memory=True)
                encoders.append(enc)
            except NotImplementedError as e:
                print(f"  [SKIP] {name}: {e}")

        print(f"\nCross-encoder CKA: {[e.name for e in encoders]}")
        result = cka_cross_encoder(
            encoders, loaders, args.dataset, out_dir,
            max_samples=args.max_samples,
        )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Vision Encoder Testbed")
    parser.add_argument("--encoder", type=str, nargs="+", default=["dinov2"],
                        help=f"Encoder name(s) or 'all'. Available: {', '.join(list_encoders())}")
    parser.add_argument("--task", type=str, default="knn", choices=list(TASKS),
                        help="Task to run")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name (cifar10, stl10, flowers102, food101, oxfordpets)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not set)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for datasets")
    # linear_probe args
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for linear probe")
    # tsne args
    parser.add_argument("--layer", type=str, default=None,
                        help="Extract from a specific layer (e.g. 'blocks.5', 'layer3'), or 'all' for every block")
    parser.add_argument("--list-layers", action="store_true",
                        help="Print available layer names for the encoder(s) and exit")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples for t-SNE")
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity")
    # color_tsne args
    parser.add_argument("--color-space", type=str, default="rgb", choices=["rgb", "hsv"],
                        help="Color space for color_tsne sampling (rgb or hsv)")
    parser.add_argument("--reduction", type=str, default="tsne", choices=["tsne", "pca"],
                        help="Dimensionality reduction method for color_tsne")
    # masking args
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Patch size for masking (pixels)")
    # reconstruction args
    parser.add_argument("--num-images", type=int, default=8,
                        help="Number of images for reconstruction visualization")
    parser.add_argument("--mask-ratio", type=float, default=0.75,
                        help="Mask ratio for MAE reconstruction")
    # finetune args
    parser.add_argument("--ft-lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--ft-epochs", type=int, default=20, help="Epochs for fine-tuning")
    # cka args
    parser.add_argument("--layers", type=str, nargs="+", default=None,
                        help="Layers for cross-layer CKA (e.g. blocks.0 blocks.3 blocks.6 blocks.11)")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    # Determine which encoders to run
    if "all" in args.encoder:
        encoder_names = list_encoders()
    else:
        encoder_names = args.encoder

    # --list-layers mode
    if args.list_layers:
        for enc_name in encoder_names:
            try:
                encoder = get_encoder(enc_name, device=args.device)
                layers = encoder.list_layers()
                print(f"\n{encoder.name} — {len(layers)} layers:")
                for l in layers:
                    print(f"  {l}")
            except NotImplementedError as e:
                print(f"\n  [SKIP] {enc_name}: {e}")
        return

    # Multi-encoder tasks — handle separately
    if args.task == "cka":
        _run_cka(encoder_names, args)
        return

    if args.task == "attention_masking":
        out_dir = OUTPUT_ROOT / "attention_masking"
        start = time.time()
        result = attention_masking_visualize(
            encoder_names, args.dataset, out_dir,
            device=args.device,
            mask_ratios=[0.0, 0.25, 0.5, 0.75],
            image_indices=list(range(args.num_images)),
            data_root=args.data_root,
        )
        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f}s — saved {len(result.get('plots', []))} figures")
        return

    if args.task == "clip_alignment":
        out_dir = OUTPUT_ROOT / "clip_alignment"
        start = time.time()
        result = clip_masking_alignment(
            args.dataset, out_dir,
            device=args.device,
            num_images=args.num_images,
            data_root=args.data_root,
        )
        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f}s")
        return

    # Expand --layer all: detect block layers per encoder and iterate
    all_results = []
    for name in encoder_names:
        try:
            if args.layer == "all":
                encoder = get_encoder(name, device=args.device)
                layer_list = _detect_block_layers(encoder)
                print(f"\n{encoder.name}: auto-detected {len(layer_list)} layers")
                del encoder  # free memory, run_single will reload
                for layer in layer_list:
                    result = run_single(name, args.task, args, layer=layer)
                    all_results.append(result)
                    if "accuracy" in result:
                        print(f"  -> {result['encoder']} @ {layer}: accuracy={result['accuracy']:.4f} ({result['time_sec']}s)")
                    else:
                        print(f"  -> {result['encoder']} @ {layer}: done ({result['time_sec']}s)")
            else:
                result = run_single(name, args.task, args, layer=args.layer)
                all_results.append(result)
                if "accuracy" in result:
                    print(f"  -> {result['encoder']}: accuracy={result['accuracy']:.4f} ({result['time_sec']}s)")
                else:
                    print(f"  -> {result['encoder']}: done ({result['time_sec']}s)")
        except NotImplementedError as e:
            print(f"\n  [SKIP] {name}: {e}")
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}", file=sys.stderr)

    if len(all_results) > 1:
        print_results_table(all_results, args.task)
    elif len(all_results) == 1 and "accuracy" in all_results[0]:
        r = all_results[0]
        print(f"\nFinal: {r['encoder']} accuracy = {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
