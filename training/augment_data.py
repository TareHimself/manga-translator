#!/usr/bin/env python3
"""
Offline, label-aware Albumentations augmentation for a small YOLO
bounding-box dataset (speech-bubble detection).

Takes an existing YOLO dataset (a data.yaml, or a directory containing one)
and produces a new, augmented dataset directory. Train/val split, class
count, and class names are all read from the source data.yaml -- nothing
is re-specified or re-split. Only the train split is augmented; val (and
test, if present) are copied through untouched so your metrics stay honest.

Every geometric transform (affine, perspective, crop) is applied to the
image AND its bounding boxes together, so boxes stay correct.

Usage:
    python augment_dataset.py \
        --dataset path/to/data.yaml \
        --output-dir dataset_augmented \
        --augs-per-image 20

    # or point at the dataset folder directly if it holds exactly one *.yaml
    python augment_dataset.py --dataset path/to/dataset --output-dir dataset_augmented

Produces:
    dataset_augmented/
        images/train, images/val[, images/test]
        labels/train, labels/val[, labels/test]
        preview/            <- boxes drawn on augmented images, ALWAYS check these
        data.yaml
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------
# dataset.yaml resolution
# --------------------------------------------------------------------------

def resolve_yaml_path(dataset_arg: Path) -> Path:
    if dataset_arg.is_file():
        return dataset_arg
    if dataset_arg.is_dir():
        candidates = sorted(dataset_arg.glob("*.yaml")) + sorted(dataset_arg.glob("*.yml"))
        named = [c for c in candidates if c.name in ("data.yaml", "dataset.yaml")]
        if named:
            return named[0]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise SystemExit(
                f"Multiple yaml files in {dataset_arg}, pass the one to use directly: "
                f"{[c.name for c in candidates]}"
            )
    raise SystemExit(f"Could not find a data.yaml at or inside {dataset_arg}")


def image_to_label_path(img_path: Path) -> Path:
    """Mirrors Ultralytics' own images/ -> labels/ path convention."""
    s = str(img_path)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa in s:
        s = sb.join(s.rsplit(sa, 1))
        return Path(s).with_suffix(".txt")
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")  # last "images" component
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    raise SystemExit(
        f"Can't infer a labels/ path for {img_path} (no 'images' folder in its path). "
        f"Rename your images directory to 'images' to match the YOLO/Ultralytics convention."
    )


def resolve_split_images(base_dir: Path, split_value) -> list[Path]:
    if split_value is None:
        return []
    if isinstance(split_value, list):
        out = []
        for v in split_value:
            out.extend(resolve_split_images(base_dir, v))
        return out
    p = Path(split_value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    if p.is_file() and p.suffix.lower() == ".txt":
        paths = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            lp = Path(line)
            if not lp.is_absolute():
                lp = (base_dir / lp).resolve()
            paths.append(lp)
        return paths
    if p.is_dir():
        return sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS)
    if p.is_file():
        return [p]
    raise SystemExit(f"Could not resolve dataset split path: {p}")


def load_dataset(yaml_path: Path):
    cfg = yaml.safe_load(yaml_path.read_text())
    base = cfg.get("path")
    if base:
        base_dir = Path(base)
        if not base_dir.is_absolute():
            base_dir = (yaml_path.parent / base_dir).resolve()
    else:
        base_dir = yaml_path.parent.resolve()

    splits = {}
    for split in ("train", "val", "test"):
        imgs = resolve_split_images(base_dir, cfg.get(split))
        pairs = []
        for img_path in imgs:
            label_path = image_to_label_path(img_path)
            if not label_path.exists():
                print(f"WARNING: no label for {img_path}, skipping")
                continue
            pairs.append((img_path, label_path))
        splits[split] = pairs

    return cfg, splits


# --------------------------------------------------------------------------
# YOLO label io
# --------------------------------------------------------------------------

def read_yolo_label(label_path: Path) -> tuple[list[list[float]], list[int]]:
    bboxes, class_ids = [], []
    for line in label_path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:5])
        bboxes.append([x, y, w, h])
        class_ids.append(cls)
    return bboxes, class_ids


def write_yolo_label(label_path: Path, bboxes, class_ids) -> None:
    lines = [
        f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for (x, y, w, h), cls in zip(bboxes, class_ids)
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def build_train_transform(height: int, width: int, border_fill: int) -> A.Compose:
    fill = (float(border_fill),) * 3
    return A.Compose(
        [
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.04),
                rotate=(-6, 6),
                shear=(-3, 3),
                fit_output=False,
                border_mode=cv2.BORDER_CONSTANT,
                fill=fill,
                p=0.85,
            ),
            A.Perspective(
                scale=(0.02, 0.05),
                fit_output=False,
                border_mode=cv2.BORDER_CONSTANT,
                fill=fill,
                p=0.3,
            ),
            A.HorizontalFlip(p=0.5),
            # erosion_rate=0.0 -> the crop region is chosen to fully contain every
            # box that's kept; it is not allowed to slice through one.
            A.RandomSizedBBoxSafeCrop(
                height=height, width=width, erosion_rate=0.0, p=0.4
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_range=(3, 5), p=1.0),
                    A.GaussianBlur(blur_range=(3, 5), p=1.0),
                ],
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.RandomBrightnessContrast(brightness_range=(-0.25, 0.25), contrast_range=(-0.25, 0.25), p=0.6),
            A.CLAHE(clip_range=(1.0, 2.0), p=0.2),
            A.ToGray(p=0.15),
            A.ImageCompression(quality_range=(35, 85), p=0.35),
            A.Downscale(scale_range=(0.5, 0.9), p=0.2),
            # RandomShadow and CoarseDropout deliberately removed: both leave a box's
            # coordinates untouched while hiding part of what's inside it, which is
            # exactly the "obstructed" case this pipeline must not produce.
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",
            label_fields=["class_labels"],
            # A box must keep essentially all of its original area or it's dropped
            # rather than kept partially clipped. Not a literal 1.0: the yolo-format
            # pixel<->normalized round trip alone introduces ~1e-5 float noise, so an
            # exact 1.0 threshold rejects every box, including ones nothing touched.
            min_visibility=0.999,
            clip_bboxes_on_input=True,
            filter_invalid_bboxes=True,
            check_each_transform=True,
        ),
    )


def draw_preview(image_bgr: np.ndarray, bboxes, out_path: Path) -> None:
    h, w = image_bgr.shape[:2]
    vis = image_bgr.copy()
    for x, y, bw, bh in bboxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(out_path), vis)


def augment_once(transform: A.Compose, image, bboxes, class_ids, max_retries: int = 40):
    """Retries until every original box survives fully intact (see min_visibility=1.0
    in build_train_transform) with at least one box kept. A rejected attempt means
    some box got clipped or eroded by that draw -- not a partial result, no result."""
    n_original = len(bboxes)
    for _ in range(max_retries):
        result = transform(image=image, bboxes=bboxes, class_labels=class_ids)
        if len(result["bboxes"]) == n_original:
            return result
    return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def copy_split_through(pairs, out: Path, split: str) -> None:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img_path, label_path in pairs:
        shutil.copy2(img_path, out / "images" / split / img_path.name)
        shutil.copy2(label_path, out / "labels" / split / f"{img_path.stem}.txt")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset", required=True, type=Path,
        help="Path to data.yaml, or a directory containing exactly one *.yaml",
    )
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--augs-per-image", type=int, default=20)
    ap.add_argument(
        "--border-fill", type=int, default=255,
        help="Pixel value for canvas padding revealed by rotate/crop: 255=white page, 0=black",
    )
    ap.add_argument("--num-preview", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    yaml_path = resolve_yaml_path(args.dataset)
    cfg, splits = load_dataset(yaml_path)
    train_pairs, val_pairs, test_pairs = splits["train"], splits["val"], splits["test"]

    if not train_pairs:
        raise SystemExit(f"No train images resolved from {yaml_path} (train: {cfg.get('train')!r})")

    train_set = {p.resolve() for p, _ in train_pairs}
    val_set = {p.resolve() for p, _ in val_pairs}
    overlap = train_set & val_set
    if overlap:
        print(
            f"WARNING: {len(overlap)} image(s) appear in BOTH train and val in {yaml_path}. "
            f"Augmented copies are only added to train, but the underlying overlap already "
            f"means val isn't a clean held-out set -- worth fixing at the source."
        )

    print(f"Source: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test images")

    out = args.output_dir
    copy_split_through(val_pairs, out, "val")
    copy_split_through(test_pairs, out, "test")

    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    preview_dir = out / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_budget = args.num_preview
    total_written = 0
    for img_path, label_path in train_pairs:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        bboxes, class_ids = read_yolo_label(label_path)

        # keep the untouched original in the training set too
        shutil.copy2(img_path, out / "images" / "train" / img_path.name)
        shutil.copy2(label_path, out / "labels" / "train" / f"{img_path.stem}.txt")
        total_written += 1

        if not bboxes:
            print(f"WARNING: {img_path.name} has no boxes, only copying the original")
            continue

        transform = build_train_transform(h, w, args.border_fill)
        made = 0
        for i in range(args.augs_per_image):
            result = augment_once(transform, image, bboxes, class_ids)
            if result is None:
                continue  # couldn't find a draw with zero clipping within the retry budget
            aug_img_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            aug_bboxes = result["bboxes"]
            aug_classes = result["class_labels"]

            stem = f"{img_path.stem}_aug{i:03d}"
            out_img_path = out / "images" / "train" / f"{stem}{img_path.suffix}"
            out_lbl_path = out / "labels" / "train" / f"{stem}.txt"
            cv2.imwrite(str(out_img_path), aug_img_bgr)
            write_yolo_label(out_lbl_path, aug_bboxes, aug_classes)
            total_written += 1
            made += 1

            if preview_budget > 0:
                draw_preview(aug_img_bgr, aug_bboxes, preview_dir / f"{stem}_preview.jpg")
                preview_budget -= 1

        if made < args.augs_per_image:
            print(
                f"{img_path.name}: only {made}/{args.augs_per_image} augmented samples found "
                f"a fully-unclipped draw within {40} retries each (boxes close to the page edge "
                f"in the source image make a clean draw harder to find)."
            )

    print(f"Wrote {total_written} training images total (originals + augmented)")

    names = cfg.get("names")
    if names is None:
        all_pairs = train_pairs + val_pairs + test_pairs
        max_cls = max(
            (c for _, lp in all_pairs for c in read_yolo_label(lp)[1]), default=0
        )
        names = {i: f"class{i}" for i in range(max_cls + 1)}
        print(f"WARNING: {yaml_path} has no 'names' key, inferred {len(names)} class(es) from label files")
    elif isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    nc = cfg.get("nc", len(names))

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val" if val_pairs else "images/train",
        "nc": nc,
        "names": names,
    }
    if test_pairs:
        data_yaml["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"Wrote {out / 'data.yaml'}")
    print(f"Sanity-check boxes in {preview_dir} before you train.")


if __name__ == "__main__":
    main()
