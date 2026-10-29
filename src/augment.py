"""Offline, bounding-box-preserving data augmentation.

Reads a manifest, writes N augmented copies of each selected image (by default
only the *train* split, so validation/test stay untouched) and writes a new
manifest containing the original records plus the augmented ones. Every
geometric transform is applied jointly to the image and its boxes via
albumentations, so boxes stay aligned; boxes that become mostly invisible are
removed (``--min_visibility``).

Augmented copies inherit ``sample_id`` and ``split`` from their source image,
so grouped splits and paired statistics remain valid.

Modality-specific presets:

* microscopic: exact flips/90-degree rotations, mild affine, brightness/contrast,
  blur/noise, CLAHE. Hue is NOT shifted: in polarised light / dispersion
  staining microscopy the colours are diagnostic of the fibre type.
* smartphone: flips, affine, perspective, lighting/colour/gamma jitter, motion
  blur, JPEG compression - i.e. realistic phone-capture variation.

Example::

    python -m src.augment --data_dir $DATA --copies 4 --balance \
        --output manifest_aug.jsonl --preview 20
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.labels import add_label_args, label_space_from_args
from src.manifest import read_manifest, resolve_image, summarise, write_manifest

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger("augment")


def build_pipeline(modality: str, min_visibility: float, strength: float = 1.0) -> A.Compose:
    """Albumentations pipeline that transforms image and Pascal VOC boxes together."""
    s = strength
    if modality == "microscopic":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            # 'ellipse' keeps boxes tight for rotated elongated objects (fibres).
            A.Affine(scale=(1 - 0.1 * s, 1 + 0.1 * s), rotate=(-15 * s, 15 * s),
                     rotate_method="ellipse", p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15 * s, contrast_limit=0.15 * s, p=0.6),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=int(10 * s),
                                 val_shift_limit=int(10 * s), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.GaussNoise(p=0.2),
        ]
    elif modality == "smartphone":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.25),
            A.Affine(scale=(1 - 0.15 * s, 1 + 0.15 * s), rotate=(-20 * s, 20 * s),
                     translate_percent=(-0.05 * s, 0.05 * s), rotate_method="ellipse", p=0.5),
            A.Perspective(scale=(0.02, 0.02 + 0.04 * s), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.25 * s, contrast_limit=0.25 * s, p=0.7),
            A.HueSaturationValue(hue_shift_limit=int(5 * s), sat_shift_limit=int(20 * s),
                                 val_shift_limit=int(15 * s), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=(3, 5))], p=0.25),
            A.ImageCompression(p=0.3),
            A.GaussNoise(p=0.2),
        ]
    else:
        raise ValueError(f"Unknown modality {modality}")
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["box_labels"],
                                 min_visibility=min_visibility, min_area=4.0),
    )


def augment_record(
    rec: dict,
    image: np.ndarray,
    pipeline: A.Compose,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Apply the pipeline deterministically for a given seed."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if hasattr(pipeline, "set_random_seed"):  # albumentations >= 1.4.21 uses its own generators
        pipeline.set_random_seed(seed % (2**32))
    h, w = image.shape[:2]
    bboxes, labels = [], []
    for b in rec.get("boxes", []):
        x0, y0, x1, y1 = b["bbox"]
        x0, x1 = sorted((max(0.0, x0), min(float(w), x1)))
        y0, y1 = sorted((max(0.0, y0), min(float(h), y1)))
        if x1 - x0 >= 1 and y1 - y0 >= 1:
            bboxes.append([x0, y0, x1, y1])
            labels.append(b.get("label", rec["label"]))
    out = pipeline(image=image, bboxes=bboxes, box_labels=labels)
    boxes = [{"bbox": [round(float(v), 2) for v in bb], "label": lab}
             for bb, lab in zip(out["bboxes"], out["box_labels"])]
    return out["image"], boxes


def draw_preview(image: np.ndarray, boxes: list[dict], path: Path) -> None:
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    thick = max(2, int(round(min(vis.shape[:2]) / 400)))
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in b["bbox"])
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), thick)
        cv2.putText(vis, str(b["label"]), (x0, max(0, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * thick, (0, 0, 255), max(1, thick // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def copies_per_record(records: list[dict], keys: list[str], copies: int, balance: bool) -> list[int]:
    """Number of augmented copies per record.

    With ``balance`` each (modality, class) group is topped up towards the size
    of the largest class of that modality *after* adding ``copies`` copies of
    it, so minority classes receive proportionally more synthetic images.
    """
    if not balance:
        return [copies] * len(records)
    counts = Counter((r["modality"], k) for r, k in zip(records, keys))
    target = {}
    for (mod, _), _n in counts.items():
        target[mod] = max(n for (m, _), n in counts.items() if m == mod) * (copies + 1)
    out = []
    for r, k in zip(records, keys):
        n = counts[(r["modality"], k)]
        out.append(max(copies, math.ceil(target[r["modality"]] / n) - 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest", default=None, help="Default: <data_dir>/manifest.jsonl")
    ap.add_argument("--output", default="manifest_aug.jsonl", help="Output manifest (relative to data_dir).")
    ap.add_argument("--out_subdir", default="augmented", help="Augmented images go to <data_dir>/<out_subdir>/")
    ap.add_argument("--splits", nargs="+", default=["train"], help="Splits to augment (never augment test).")
    ap.add_argument("--modality", choices=["microscopic", "smartphone", "unified"], default="unified")
    ap.add_argument("--copies", type=int, default=3, help="Augmented copies per image (minimum).")
    ap.add_argument("--balance", action="store_true", help="Generate extra copies for minority classes.")
    ap.add_argument("--max_copies", type=int, default=20, help="Upper bound per image when balancing.")
    ap.add_argument("--strength", type=float, default=1.0, help="Scales augmentation magnitudes.")
    ap.add_argument("--min_visibility", type=float, default=0.4,
                    help="Drop a transformed box if less than this fraction of it stays visible.")
    ap.add_argument("--require_boxes", action="store_true",
                    help="Discard an augmented image if it had boxes and all were lost.")
    ap.add_argument("--preview", type=int, default=0, help="Write N box-overlay previews for checking.")
    ap.add_argument("--seed", type=int, default=1234)
    add_label_args(ap)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if "test" in args.splits:
        log.warning("Augmenting the test split inflates and correlates test samples - not recommended.")

    data_dir = Path(args.data_dir)
    manifest = Path(args.manifest) if args.manifest else data_dir / "manifest.jsonl"
    records = read_manifest(manifest)
    originals = [r for r in records if "augmented_from" not in r]
    space = label_space_from_args(args)

    selected, keys = [], []
    for r in originals:
        if r.get("split") not in args.splits:
            continue
        if args.modality != "unified" and r["modality"] != args.modality:
            continue
        key = space.map(r["label"])
        if key is None:
            continue
        selected.append(r)
        keys.append(key)
    n_copies = [min(c, args.max_copies) for c in copies_per_record(selected, keys, args.copies, args.balance)]
    log.info("Augmenting %d images -> %d new images", len(selected), sum(n_copies))

    pipelines = {m: build_pipeline(m, args.min_visibility, args.strength) for m in ("microscopic", "smartphone")}
    out_dir = data_dir / args.out_subdir
    preview_dir = out_dir / "_preview"
    new_records, lost_boxes, discarded, previews = [], 0, 0, 0

    for idx, (rec, n) in enumerate(tqdm(list(zip(selected, n_copies)), desc="augment")):
        if n <= 0:
            continue
        with Image.open(resolve_image(rec, data_dir)) as im:
            image = np.asarray(im.convert("RGB"))
        src = Path(rec["image"])
        lossless = src.suffix.lower() in (".png", ".tif", ".tiff", ".bmp")
        for k in range(n):
            seed = args.seed * 1_000_003 + idx * 1_009 + k
            aug, boxes = augment_record(rec, image, pipelines[rec["modality"]], seed)
            had = len(rec.get("boxes", []))
            lost_boxes += had - len(boxes)
            if args.require_boxes and had and not boxes:
                discarded += 1
                continue
            rel = Path(args.out_subdir) / rec["modality"] / src.parent.name / f"{src.stem}__aug{k:02d}{'.png' if lossless else '.jpg'}"
            dst = data_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            pil = Image.fromarray(aug)
            if lossless:
                pil.save(dst)
            else:
                pil.save(dst, quality=95)
            new = {key: v for key, v in rec.items() if key not in ("boxes", "image", "id", "width", "height")}
            new.update({
                "id": f"{rec['id']}__aug{k:02d}",
                "image": rel.as_posix(),
                "width": int(aug.shape[1]),
                "height": int(aug.shape[0]),
                "boxes": boxes,
                "augmented_from": rec["id"],
            })
            new_records.append(new)
            if previews < args.preview:
                draw_preview(aug, boxes, preview_dir / f"{rec['id'].replace('/', '_')}__aug{k:02d}.jpg")
                if k == 0:
                    draw_preview(image, rec.get("boxes", []), preview_dir / f"{rec['id'].replace('/', '_')}__orig.jpg")
                previews += 1

    out_path = Path(args.output)
    out_path = out_path if out_path.is_absolute() else data_dir / out_path
    write_manifest(originals + new_records, out_path)
    log.info("Wrote %s: %d original + %d augmented records", out_path, len(originals), len(new_records))
    log.info("Boxes removed for low visibility: %d; images discarded: %d", lost_boxes, discarded)
    log.info("Counts after augmentation (raw labels):\n%s",
             summarise(originals + new_records))
    if args.preview:
        log.info("Box previews written to %s - inspect them before training.", preview_dir)


if __name__ == "__main__":
    main()
