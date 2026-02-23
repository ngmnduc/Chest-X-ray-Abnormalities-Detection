"""
prepare_data.py
===============
Converts the raw VinDr-CXR annotations (CSV) into YOLO format label files and
organises the 512×512 PNG images into train / val / test splits.

All 14 original finding categories are collapsed into a single "Abnormality"
class (class index 0).  Images that contain no annotated findings are treated
as negative samples and receive an empty label file.

Expected raw dataset layout
---------------------------
<raw_dir>/
    train.csv          # training annotations
    test.csv           # public test annotations  (optional)
    train/             # 512×512 PNG images (image_id.png)
    test/              # 512×512 PNG images (optional)

Output layout (YOLO)
--------------------
<out_dir>/
    images/
        train/   val/   test/
    labels/
        train/   val/   test/

Usage
-----
    python src/prepare_data.py \\
        --raw_dir  /path/to/vindr-cxr-raw \\
        --out_dir  data/vindr_yolo \\
        --val_frac 0.1 \\
        --test_frac 0.1 \\
        --seed 42
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# All VinDr-CXR finding class names (14 categories → class 0 "Abnormality")
VINDR_CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

IMAGE_SIZE = 512  # VinDr-CXR images are 512×512 pixels


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV annotations and normalise column names."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def convert_to_yolo(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_w: int = IMAGE_SIZE,
    img_h: int = IMAGE_SIZE,
) -> tuple:
    """Convert absolute (x_min, y_min, x_max, y_max) to YOLO format.

    Returns
    -------
    tuple: (x_center, y_center, width, height) all normalised to [0, 1].
    """
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    # Clamp to [0, 1] to guard against annotation noise
    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    width = float(np.clip(width, 0.0, 1.0))
    height = float(np.clip(height, 0.0, 1.0))
    return x_center, y_center, width, height


def build_label_lines(group: pd.DataFrame) -> list:
    """Build YOLO label lines for one image's annotations.

    Only rows that carry valid bounding boxes are included.  Rows where the
    class name is 'No finding' (i.e. negative images) are skipped so the
    label file is left empty.

    Parameters
    ----------
    group : pd.DataFrame
        Subset of the annotation DataFrame for a single image_id.

    Returns
    -------
    list of str: One YOLO label line per valid bounding box.
    """
    lines = []
    for _, row in group.iterrows():
        class_name = str(row.get("class_name", "")).strip()
        if class_name.lower() == "no finding":
            continue
        try:
            x_min = float(row["x_min"])
            y_min = float(row["y_min"])
            x_max = float(row["x_max"])
            y_max = float(row["y_max"])
        except (KeyError, ValueError):
            continue
        # Skip degenerate boxes
        if x_max <= x_min or y_max <= y_min:
            continue
        xc, yc, w, h = convert_to_yolo(x_min, y_min, x_max, y_max)
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def split_image_ids(
    image_ids: list,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple:
    """Split image IDs into train / val / test sets.

    Parameters
    ----------
    image_ids : list
    val_frac : float  Fraction of total for validation.
    test_frac : float Fraction of total for test.
    seed : int        Random seed for reproducibility.

    Returns
    -------
    tuple: (train_ids, val_ids, test_ids) as lists.
    """
    test_size = test_frac / (1.0 - val_frac) if (1.0 - val_frac) > 0 else 0.0
    train_val, test_ids = train_test_split(
        image_ids, test_size=test_frac, random_state=seed
    )
    val_size = val_frac / (1.0 - test_frac) if (1.0 - test_frac) > 0 else 0.0
    train_ids, val_ids = train_test_split(
        train_val, test_size=val_size, random_state=seed
    )
    return train_ids, val_ids, test_ids


def prepare_split(
    image_ids: list,
    split_name: str,
    annotations: dict,
    src_image_dir: Path,
    out_dir: Path,
    copy_images: bool,
) -> None:
    """Copy images and write label files for one split.

    Parameters
    ----------
    image_ids : list            Image IDs in this split.
    split_name : str            'train', 'val', or 'test'.
    annotations : dict          image_id → list of YOLO label lines.
    src_image_dir : Path        Directory containing source PNG files.
    out_dir : Path              Root output directory.
    copy_images : bool          Whether to copy image files.
    """
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for image_id in tqdm(image_ids, desc=f"Preparing {split_name}", unit="img"):
        # Write label file
        lines = annotations.get(image_id, [])
        label_path = lbl_out / f"{image_id}.txt"
        label_path.write_text("\n".join(lines))

        # Copy image if available
        if copy_images and src_image_dir.exists():
            src = src_image_dir / f"{image_id}.png"
            if src.exists():
                shutil.copy2(src, img_out / f"{image_id}.png")


def prepare_dataset(
    raw_dir: Path,
    out_dir: Path,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    copy_images: bool = True,
) -> dict:
    """Main entry point: prepare the YOLO dataset from raw VinDr-CXR files.

    Parameters
    ----------
    raw_dir : Path    Directory containing train.csv and train/ images.
    out_dir : Path    Destination directory for the YOLO dataset.
    val_frac : float  Fraction of training data for validation.
    test_frac : float Fraction of training data for test.
    seed : int        Random seed.
    copy_images : bool  Set to False to only write label files (faster for CI).

    Returns
    -------
    dict with keys 'train', 'val', 'test' mapping to lists of image IDs.
    """
    train_csv = raw_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Annotation file not found: {train_csv}")

    df = load_annotations(train_csv)

    # Build per-image annotation dict
    annotations: dict = {}
    for image_id, group in df.groupby("image_id"):
        annotations[image_id] = build_label_lines(group)

    all_ids = sorted(annotations.keys())
    train_ids, val_ids, test_ids = split_image_ids(
        all_ids, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    src_image_dir = raw_dir / "train"
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        prepare_split(
            image_ids=ids,
            split_name=split_name,
            annotations=annotations,
            src_image_dir=src_image_dir,
            out_dir=out_dir,
            copy_images=copy_images,
        )

    stats = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    print(
        f"\nDataset prepared at: {out_dir}\n"
        f"  train: {len(train_ids)} images\n"
        f"  val:   {len(val_ids)} images\n"
        f"  test:  {len(test_ids)} images"
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare VinDr-CXR data in YOLO format."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Path to the raw VinDr-CXR directory containing train.csv and train/.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/vindr_yolo"),
        help="Output directory for the YOLO-formatted dataset.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1).",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no_copy_images",
        action="store_true",
        help="Skip copying image files (only write label files).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        copy_images=not args.no_copy_images,
    )
