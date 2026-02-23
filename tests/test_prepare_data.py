"""
test_prepare_data.py
====================
Unit tests for the data preparation module (src/prepare_data.py).

Tests validate:
  - YOLO bounding-box conversion correctness
  - Edge cases (degenerate boxes, 'No finding' labels, clamping)
  - Label line generation
  - Train/val/test split proportions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prepare_data import (
    build_label_lines,
    convert_to_yolo,
    load_annotations,
    split_image_ids,
)


class TestConvertToYolo:
    """Tests for the convert_to_yolo() coordinate transformation."""

    def test_centre_box(self):
        """A box spanning the whole image should have centre (0.5, 0.5)."""
        xc, yc, w, h = convert_to_yolo(0, 0, 512, 512, img_w=512, img_h=512)
        assert xc == pytest.approx(0.5)
        assert yc == pytest.approx(0.5)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(1.0)

    def test_top_left_box(self):
        """A box in the top-left quadrant."""
        xc, yc, w, h = convert_to_yolo(0, 0, 256, 256, img_w=512, img_h=512)
        assert xc == pytest.approx(0.25)
        assert yc == pytest.approx(0.25)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)

    def test_arbitrary_box(self):
        """Arbitrary bounding box conversion."""
        xc, yc, w, h = convert_to_yolo(100, 50, 300, 400, img_w=512, img_h=512)
        assert xc == pytest.approx(200 / 512)
        assert yc == pytest.approx(225 / 512)
        assert w == pytest.approx(200 / 512)
        assert h == pytest.approx(350 / 512)

    def test_clamp_overflow(self):
        """Coordinates exceeding image bounds should be clamped to [0, 1]."""
        xc, yc, w, h = convert_to_yolo(-10, -10, 600, 600, img_w=512, img_h=512)
        assert 0.0 <= xc <= 1.0
        assert 0.0 <= yc <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0

    def test_output_range(self):
        """Random boxes should always produce values in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            coords = rng.integers(0, 513, size=4)
            x_min, y_min = coords[:2].min(), coords[2:].min()
            x_max = x_min + rng.integers(1, 512)
            y_max = y_min + rng.integers(1, 512)
            xc, yc, w, h = convert_to_yolo(x_min, y_min, x_max, y_max)
            for v in (xc, yc, w, h):
                assert 0.0 <= v <= 1.0, f"Out-of-range value {v} for box ({x_min},{y_min},{x_max},{y_max})"


class TestBuildLabelLines:
    """Tests for build_label_lines() which generates per-image YOLO labels."""

    def _make_row(self, class_name, x_min, y_min, x_max, y_max, image_id="img1"):
        return {
            "image_id": image_id,
            "class_name": class_name,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    def test_single_valid_box(self):
        df = pd.DataFrame([self._make_row("Cardiomegaly", 10, 20, 100, 150)])
        lines = build_label_lines(df)
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0", "class index must be 0 (Abnormality)"
        assert len(parts) == 5

    def test_no_finding_skipped(self):
        df = pd.DataFrame([self._make_row("No finding", 0, 0, 512, 512)])
        lines = build_label_lines(df)
        assert lines == [], "No finding rows should produce no label lines"

    def test_mixed_classes_all_become_class_0(self):
        rows = [
            self._make_row("Aortic enlargement", 10, 10, 100, 100),
            self._make_row("Nodule/Mass", 200, 200, 300, 300),
            self._make_row("No finding", 0, 0, 512, 512),
        ]
        df = pd.DataFrame(rows)
        lines = build_label_lines(df)
        assert len(lines) == 2
        for line in lines:
            assert line.startswith("0 "), "All classes should map to class 0"

    def test_degenerate_box_skipped(self):
        """Boxes where x_max <= x_min or y_max <= y_min should be dropped."""
        rows = [
            self._make_row("Cardiomegaly", 100, 100, 100, 200),  # zero width
            self._make_row("Cardiomegaly", 100, 200, 200, 200),  # zero height
            self._make_row("Cardiomegaly", 100, 100, 200, 200),  # valid
        ]
        df = pd.DataFrame(rows)
        lines = build_label_lines(df)
        assert len(lines) == 1

    def test_label_format(self):
        """Each label line must have exactly 5 space-separated tokens."""
        df = pd.DataFrame([self._make_row("Consolidation", 50, 50, 200, 300)])
        lines = build_label_lines(df)
        assert len(lines) == 1
        parts = lines[0].split()
        assert len(parts) == 5
        # All values after class index must be floats in [0, 1]
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0


class TestSplitImageIds:
    """Tests for split_image_ids() – train/val/test proportions."""

    def test_split_sizes(self):
        ids = [f"img{i:04d}" for i in range(1000)]
        train, val, test = split_image_ids(ids, val_frac=0.1, test_frac=0.1, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == 1000
        assert abs(len(val) - 100) <= 2
        assert abs(len(test) - 100) <= 2

    def test_no_overlap(self):
        ids = [f"img{i:04d}" for i in range(500)]
        train, val, test = split_image_ids(ids, val_frac=0.1, test_frac=0.1, seed=0)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_reproducibility(self):
        ids = [f"img{i:04d}" for i in range(200)]
        t1, v1, te1 = split_image_ids(ids, val_frac=0.1, test_frac=0.1, seed=7)
        t2, v2, te2 = split_image_ids(ids, val_frac=0.1, test_frac=0.1, seed=7)
        assert t1 == t2
        assert v1 == v2
        assert te1 == te2


class TestLoadAnnotations:
    """Tests for load_annotations() CSV loading."""

    def test_load_csv(self, tmp_path):
        csv_content = (
            "image_id,class_name,class_id,x_min,y_min,x_max,y_max\n"
            "abc123,Cardiomegaly,3,10,20,100,200\n"
            "abc123,No finding,14,,,,"
        )
        csv_path = tmp_path / "train.csv"
        csv_path.write_text(csv_content)
        df = load_annotations(csv_path)
        assert "image_id" in df.columns
        assert "class_name" in df.columns
        assert len(df) == 2

    def test_column_normalisation(self, tmp_path):
        """Column names with mixed case and spaces should be normalised."""
        csv_content = "Image ID,Class Name,X Min,Y Min,X Max,Y Max\nid1,Cardiomegaly,10,20,100,200\n"
        csv_path = tmp_path / "train.csv"
        csv_path.write_text(csv_content)
        df = load_annotations(csv_path)
        assert "image_id" in df.columns
        assert "class_name" in df.columns
        assert "x_min" in df.columns
