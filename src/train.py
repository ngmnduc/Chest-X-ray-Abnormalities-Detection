"""
train.py
========
Train a YOLOv8 object detection model on the prepared VinDr-CXR dataset.

All 14 original finding categories are merged into a single "Abnormality"
class.  Key hyperparameters can be tuned via command-line arguments.

Usage
-----
    python src/train.py \\
        --data   data/dataset.yaml \\
        --model  yolov8n.pt \\
        --epochs 50 \\
        --imgsz  512 \\
        --batch  16 \\
        --lr0    0.01 \\
        --project runs/train \\
        --name   vindr_yolov8
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on the VinDr-CXR dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/dataset.yaml"),
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model variant (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Input image size in pixels (default: 512).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16; use -1 for auto-batch).",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate fraction of lr0 (default: 0.01).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum / Adam beta1 (default: 0.937).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Optimizer weight decay (default: 0.0005).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience in epochs (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker threads (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Training device: '' (auto), '0', 'cpu', etc.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="vindr_yolov8",
        help="Run sub-directory name.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet-pretrained weights (default: use pretrained).",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Run YOLOv8 training with the given hyperparameters."""
    # Import here so the module is importable without ultralytics installed
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        workers=args.workers,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        pretrained=not args.no_pretrained,
        exist_ok=True,
    )
    print(f"\nTraining complete. Results saved to: {results.save_dir}")
    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
