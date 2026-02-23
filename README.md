# Chest X-ray Abnormalities Detection

Detect chest X-ray abnormalities with **YOLOv8** trained on the
[VinDr-CXR](https://vindr.ai/datasets/cxr) dataset.  All 14 original finding
categories are collapsed into a single **Abnormality** class, making this a
binary object-detection task (abnormal vs. normal regions).

---

## Project structure

```
.
├── data/
│   └── dataset.yaml          # YOLO dataset configuration
├── src/
│   ├── prepare_data.py       # CSV → YOLO label conversion + train/val/test split
│   ├── train.py              # YOLOv8 training with tuneable hyperparameters
│   └── evaluate.py           # mAP / Precision / Recall + visual error analysis
├── tests/
│   └── test_prepare_data.py  # Unit tests for data preparation logic
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.9.  A CUDA-capable GPU is strongly recommended for training.

---

## 1 · Data Preparation

Download the VinDr-CXR dataset (512 × 512 PNG images + CSV annotations) from
<https://physionet.org/content/vindr-cxr/1.0.0/>.

The expected raw directory layout is:

```
<raw_dir>/
    train.csv
    test.csv        # optional
    train/          # *.png images
    test/           # *.png images (optional)
```

Run the preparation script to convert coordinates to YOLO format and create
train / val / test splits:

```bash
python src/prepare_data.py \
    --raw_dir  /path/to/vindr-cxr-raw \
    --out_dir  data/vindr_yolo \
    --val_frac 0.1 \
    --test_frac 0.1 \
    --seed 42
```

### Bounding-box conversion

Original VinDr-CXR annotations use absolute pixel coordinates
`(x_min, y_min, x_max, y_max)` on 512 × 512 images.
These are converted to normalised YOLO format as follows:

```
x_center = (x_min + x_max) / 2 / 512
y_center = (y_min + y_max) / 2 / 512
width    = (x_max - x_min) / 512
height   = (y_max - y_min) / 512
```

All 14 finding categories are mapped to **class 0** ("Abnormality").
Images annotated as "No finding" receive an empty label file (negative sample).

---

## 2 · Training

```bash
python src/train.py \
    --data    data/dataset.yaml \
    --model   yolov8n.pt \
    --epochs  50 \
    --imgsz   512 \
    --batch   16 \
    --lr0     0.01 \
    --project runs/train \
    --name    vindr_yolov8
```

Key hyperparameters:

| Argument | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | YOLOv8 variant (`n`/`s`/`m`/`l`/`x`) |
| `--epochs` | 50 | Training epochs |
| `--imgsz` | 512 | Input resolution (matches VinDr-CXR) |
| `--batch` | 16 | Batch size (`-1` for auto) |
| `--lr0` | 0.01 | Initial learning rate |
| `--lrf` | 0.01 | Final LR as a fraction of `lr0` |
| `--momentum` | 0.937 | SGD momentum / Adam β₁ |
| `--weight_decay` | 0.0005 | L2 regularisation |
| `--patience` | 10 | Early-stopping patience (epochs) |

Trained weights are saved to `runs/train/vindr_yolov8/weights/`.

---

## 3 · Evaluation

```bash
python src/evaluate.py \
    --model   runs/train/vindr_yolov8/weights/best.pt \
    --data    data/dataset.yaml \
    --split   test \
    --conf    0.25 \
    --iou     0.5 \
    --out_dir runs/evaluate
```

The script reports **mAP@0.50**, **mAP@0.50:0.95**, **Precision**, and
**Recall** using YOLOv8's built-in validator, then runs per-image inference to:

* Save up to `--max_vis` prediction visualisations (green = GT, red = prediction).
* Save all images with false positives or false negatives to `runs/evaluate/errors/`.
* Write a `metrics_summary.json` with all numeric results.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Methodology

1. **Data preparation** — Raw VinDr-CXR CSV labels are parsed, degenerate
   boxes are filtered, and all lesion categories are unified into a single
   *Abnormality* class.  The dataset is split 80 / 10 / 10 into train, val,
   and test sets with a fixed random seed for reproducibility.

2. **Model** — YOLOv8 (Ultralytics) is used for end-to-end object detection.
   The nano (`yolov8n`) or small (`yolov8s`) variant is recommended for initial
   experiments; larger variants can be used for best accuracy.

3. **Evaluation** — Standard object-detection metrics (mAP@0.50,
   mAP@0.50:0.95, Precision, Recall) are computed on the held-out test split.
   Visual error analysis highlights false positives and false negatives to guide
   further improvement.
