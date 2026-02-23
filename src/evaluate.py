"""
evaluate.py
===========
Evaluate a trained YOLOv8 model on the VinDr-CXR test split and produce:

  * Quantitative metrics – mAP50, mAP50-95, Precision, Recall
  * Visualisations   – sample predictions with bounding boxes
  * Error analysis   – false positives and false negatives saved to disk

Usage
-----
    python src/evaluate.py \\
        --model  runs/train/vindr_yolov8/weights/best.pt \\
        --data   data/dataset.yaml \\
        --split  test \\
        --conf   0.25 \\
        --iou    0.5 \\
        --out_dir runs/evaluate
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 model on VinDr-CXR and analyse errors."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained YOLOv8 weights (.pt file).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/dataset.yaml"),
        help="Path to the YOLO dataset YAML.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS and matching (default: 0.5).",
    )
    parser.add_argument(
        "--max_vis",
        type=int,
        default=20,
        help="Maximum number of images to visualise (default: 20).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/evaluate"),
        help="Output directory for evaluation artefacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Inference device: '' (auto), '0', 'cpu', etc.",
    )
    return parser.parse_args()


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format.

    Parameters
    ----------
    box_a, box_b : np.ndarray  Shape (4,) — [x1, y1, x2, y2].

    Returns
    -------
    float: Intersection-over-Union value in [0, 1].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def classify_predictions(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple:
    """Classify predicted boxes as TP, FP; count FN.

    Parameters
    ----------
    pred_boxes : np.ndarray  Shape (N, 4) — predicted boxes in xyxy format.
    gt_boxes   : np.ndarray  Shape (M, 4) — ground-truth boxes in xyxy format.
    iou_threshold : float

    Returns
    -------
    tuple: (tp, fp, fn) counts for this single image.
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = fp = 0
    for pb in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j not in matched_gt:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Convert normalised YOLO box to absolute xyxy coordinates."""
    x1 = (x_center - width / 2) * img_w
    y1 = (y_center - height / 2) * img_h
    x2 = (x_center + width / 2) * img_w
    y2 = (y_center + height / 2) * img_h
    return np.array([x1, y1, x2, y2])


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load ground-truth boxes from a YOLO label file.

    Returns
    -------
    np.ndarray: Shape (N, 4) in xyxy format; empty array if no boxes.
    """
    if not label_path.exists():
        return np.empty((0, 4))
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = float(parts[0]), *map(float, parts[1:5])
        boxes.append(yolo_to_xyxy(xc, yc, w, h, img_w, img_h))
    return np.array(boxes) if boxes else np.empty((0, 4))


def visualise_predictions(
    image_path: Path,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    out_path: Path,
) -> None:
    """Save a side-by-side figure of ground-truth vs. predictions.

    Parameters
    ----------
    image_path  : Path          Source image.
    pred_boxes  : np.ndarray    Predicted boxes in xyxy format.
    pred_scores : np.ndarray    Confidence scores for each predicted box.
    gt_boxes    : np.ndarray    Ground-truth boxes in xyxy format.
    out_path    : Path          Where to save the PNG figure.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, title in zip(axes, ["Ground Truth", "Predictions"]):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    # Ground-truth boxes (green)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        axes[0].add_patch(rect)

    # Predicted boxes (red)
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, y1 - 4, f"{score:.2f}",
            color="red", fontsize=8, fontweight="bold",
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_validation(model, data: Path, split: str, conf: float, iou: float, device: str):
    """Run YOLOv8 built-in validation and return the metrics object."""
    kwargs = dict(
        data=str(data),
        split=split,
        conf=conf,
        iou=iou,
        save_json=False,
        verbose=True,
    )
    if device:
        kwargs["device"] = device
    return model.val(**kwargs)


def evaluate(args: argparse.Namespace) -> dict:
    """Full evaluation pipeline.

    Returns
    -------
    dict: Summary of mAP50, mAP50-95, Precision, Recall and TP/FP/FN counts.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))

    # --- Quantitative metrics via built-in validation ---
    metrics = run_validation(
        model, args.data, args.split, args.conf, args.iou, args.device
    )
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    print(
        f"\n{'='*50}\n"
        f"Evaluation results ({args.split} split)\n"
        f"{'='*50}\n"
        f"  mAP@0.50:        {map50:.4f}\n"
        f"  mAP@0.50:0.95:   {map50_95:.4f}\n"
        f"  Precision:       {precision:.4f}\n"
        f"  Recall:          {recall:.4f}\n"
        f"{'='*50}"
    )

    # --- Per-image error analysis ---
    import yaml  # noqa: PLC0415

    with open(args.data) as fh:
        dataset_cfg = yaml.safe_load(fh)
    dataset_root = Path(dataset_cfg.get("path", "."))
    img_dir = dataset_root / "images" / args.split
    lbl_dir = dataset_root / "labels" / args.split

    total_tp = total_fp = total_fn = 0
    vis_count = 0
    vis_dir = args.out_dir / "visualisations"
    error_dir = args.out_dir / "errors"
    vis_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    if img_dir.exists():
        image_paths = sorted(img_dir.glob("*.png"))
        for img_path in image_paths:
            img = Image.open(img_path)
            img_w, img_h = img.size

            label_path = lbl_dir / (img_path.stem + ".txt")
            gt_boxes = load_gt_boxes(label_path, img_w, img_h)

            result = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                verbose=False,
                device=args.device if args.device else None,
            )[0]

            pred_boxes_raw = result.boxes.xyxy.cpu().numpy() if result.boxes else np.empty((0, 4))
            pred_scores_raw = result.boxes.conf.cpu().numpy() if result.boxes else np.array([])

            tp, fp, fn = classify_predictions(pred_boxes_raw, gt_boxes, args.iou)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Visualise first max_vis images
            if vis_count < args.max_vis:
                visualise_predictions(
                    image_path=img_path,
                    pred_boxes=pred_boxes_raw,
                    pred_scores=pred_scores_raw,
                    gt_boxes=gt_boxes,
                    out_path=vis_dir / f"{img_path.stem}.png",
                )
                vis_count += 1

            # Save images with errors (FP or FN)
            if fp > 0 or fn > 0:
                visualise_predictions(
                    image_path=img_path,
                    pred_boxes=pred_boxes_raw,
                    pred_scores=pred_scores_raw,
                    gt_boxes=gt_boxes,
                    out_path=error_dir / f"{img_path.stem}_FP{fp}_FN{fn}.png",
                )

    # --- Save summary to JSON ---
    summary = {
        "split": args.split,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "total_TP": total_tp,
        "total_FP": total_fp,
        "total_FN": total_fn,
    }
    summary_path = args.out_dir / "metrics_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nMetrics summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
