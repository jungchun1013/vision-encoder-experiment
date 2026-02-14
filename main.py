import argparse
import sys
import time

import torch

import models  # triggers all @register decorators
from models.registry import get_encoder, list_encoders
from datasets.loader import get_loaders, get_num_classes
from tasks.knn import knn_evaluate
from tasks.linear_probe import linear_probe_evaluate


TASKS = {
    "knn": "k-NN",
    "linear_probe": "Linear Probe",
}


def run_single(encoder_name: str, task: str, args) -> dict:
    """Run a single encoder + task combination. Returns results dict."""
    encoder = get_encoder(encoder_name, device=args.device)
    print(f"\n{'='*60}")
    print(f"Encoder: {encoder.name}  |  Task: {TASKS[task]}  |  Dataset: {args.dataset}")
    print(f"{'='*60}")

    print(f"Loading model...")
    transform = encoder.get_transform()

    print(f"Loading dataset...")
    train_loader, test_loader = get_loaders(
        args.dataset, transform, batch_size=args.batch_size, data_root=args.data_root,
    )

    num_classes = get_num_classes(args.dataset)
    start = time.time()

    if task == "knn":
        results = knn_evaluate(encoder, train_loader, test_loader, num_classes=num_classes)
    elif task == "linear_probe":
        results = linear_probe_evaluate(
            encoder, train_loader, test_loader,
            num_classes=num_classes, epochs=args.epochs,
        )

    elapsed = time.time() - start
    results["time_sec"] = round(elapsed, 1)
    results["encoder"] = encoder.name
    return results


def print_results_table(all_results: list[dict], task: str):
    """Pretty-print a results table."""
    print(f"\n{'='*60}")
    print(f"Results: {TASKS[task]}")
    print(f"{'='*60}")
    print(f"{'Encoder':<20} {'Accuracy':>10} {'Time (s)':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10}")
    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{r['encoder']:<20} {r['accuracy']:>10.4f} {r['time_sec']:>10.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Vision Encoder Testbed")
    parser.add_argument("--encoder", type=str, default="dinov2",
                        help=f"Encoder name or 'all'. Available: {', '.join(list_encoders())}")
    parser.add_argument("--task", type=str, default="knn", choices=list(TASKS),
                        help="Downstream task")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name (cifar10, stl10, flowers102, food101, oxfordpets)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for linear probe")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not set)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for datasets")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    # Determine which encoders to run
    if args.encoder == "all":
        encoder_names = list_encoders()
    else:
        encoder_names = [args.encoder]

    all_results = []
    for name in encoder_names:
        try:
            result = run_single(name, args.task, args)
            all_results.append(result)
            print(f"  -> {result['encoder']}: accuracy={result['accuracy']:.4f} ({result['time_sec']}s)")
        except NotImplementedError as e:
            print(f"\n  [SKIP] {name}: {e}")
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}", file=sys.stderr)

    if len(all_results) > 1:
        print_results_table(all_results, args.task)
    elif len(all_results) == 1:
        r = all_results[0]
        print(f"\nFinal: {r['encoder']} accuracy = {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
