"""Export a modality's bounding boxes from the manifest as a YOLO detection dataset.

The YOLO model trained on it (src/yolo_train.py) is used to pre-label new images
(src/autolabel.py); people then correct the boxes instead of drawing them all.

    python -m src.yolo_data \
        --data_dir $D --manifest $D/manifest_grouped.jsonl \
        --modality smartphone --class_config configs/classes_macro.yaml \
        --out_dir $D/yolo_macro

Layout written (Ultralytics finds each label by swapping images/ for labels/):

    out_dir/images/{train,val,test}/<name>.<ext>   links to (or copies of) the originals
    out_dir/labels/{train,val,test}/<name>.txt     "class cx cy w h", normalised
    out_dir/data.yaml                              paths and class names
    out_dir/classes.json                           class keys, display names, raw label
                                                   written back to LabelMe, asbestos flag
    out_dir/export_summary.json                    counts per split and class, problems

The splits are the manifest's, so with the rebuilt (near-duplicate grouped) manifest
the YOLO test images have no near-copy in training. Offline-augmented copies are left
out by default: Ultralytics augments on the fly, and they exist for the train split only.

Box coordinates are normalised by the image size after its EXIF orientation is applied,
which is how LabelMe shows the photo when the boxes are drawn and how Ultralytics loads it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from src.labels import add_label_args, label_space_from_args
from src.manifest import read_manifest, resolve_image

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger("yolo_data")
SPLITS = ("train", "val", "test")
CLASS_MODES = ("type", "binary", "object")
_ROTATED = {5, 6, 7, 8}          # EXIF orientations that swap width and height


def oriented_size(path: Path) -> tuple[int, int, bool]:
    """(width, height) as displayed, i.e. after the EXIF orientation is applied, and
    whether that orientation turns the stored image by 90 degrees."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            orientation = im.getexif().get(274, 1)
        except Exception:  # noqa: BLE001 - unreadable EXIF means no rotation
            orientation = 1
    turned = orientation in _ROTATED
    return (h, w, True) if turned else (w, h, False)


def safe_name(record_id: str) -> str:
    """File name for a record id that is unique across folders ("a/b/c" -> "a__b__c")."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", record_id.replace("/", "__")).strip("_")


def yolo_classes(space, mode: str, raw_of: dict[str, Counter]) -> list[dict]:
    """YOLO classes for a label space. raw_of[key] counts the raw box labels seen for a key;
    the most common one is what autolabelling writes back, so new files match old ones."""
    if mode == "object":
        return [{"key": "material", "name": "Material", "asbestos": None, "raw_label": "material"}]
    if mode == "binary":
        return [{"key": "asbestos", "name": "Asbestos", "asbestos": True, "raw_label": "asbestos"},
                {"key": "non_asbestos", "name": "Non-asbestos", "asbestos": False, "raw_label": "non-asbestos"}]
    out = []
    for c in space.classes:
        seen = raw_of.get(c.key)
        out.append({"key": c.key, "name": c.name, "asbestos": bool(c.asbestos),
                    "raw_label": seen.most_common(1)[0][0] if seen else c.key})
    return out


def class_index(space, mode: str, key: str) -> int:
    if mode == "object":
        return 0
    if mode == "binary":
        return 0 if space.is_asbestos(key) else 1
    return space.keys.index(key)


def place(src: Path, dst: Path, how: str) -> str:
    """Put src at dst by symlink, hard link or copy. Returns the method used; a link that
    the file system refuses (e.g. Windows without symlink rights) falls back to a copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if how == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return "symlink"
        except OSError:
            pass
    if how in ("symlink", "hardlink"):
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def yolo_line(cls: int, bbox: list[float], w: int, h: int) -> str | None:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, x1 = sorted((min(max(x0, 0.0), w), min(max(x1, 0.0), w)))
    y0, y1 = sorted((min(max(y0, 0.0), h), min(max(y1, 0.0), h)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def export(args) -> dict:
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    space = label_space_from_args(args, modality=args.modality)
    records = [r for r in read_manifest(args.manifest) if r.get("modality") == args.modality]
    if not args.include_augmented:
        records = [r for r in records if not r.get("augmented_from")]
    if not records:
        raise SystemExit(f"No {args.modality} records in {args.manifest}")

    raw_of: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        for b in r.get("boxes") or []:
            key = space.map(b["label"])
            if key is not None:
                raw_of[key][b["label"]] += 1
    classes = yolo_classes(space, args.class_mode, raw_of)

    for split in SPLITS:
        for sub in ("images", "labels"):
            d = out_dir / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    images = Counter()
    problems: list[dict] = []
    methods = Counter()
    for r in records:
        split = r.get("split")
        if split not in SPLITS:
            problems.append({"id": r["id"], "reason": f"split {split!r}"})
            continue
        src = resolve_image(r, data_dir)
        if not src.exists():
            problems.append({"id": r["id"], "reason": "image missing"})
            continue
        w, h, turned = oriented_size(src)
        if turned and w != h and (r.get("width"), r.get("height")) == (h, w):
            # LabelMe draws on the turned photo, but the importer clipped the boxes to the
            # stored size, which cuts off boxes that reach past the stored height.
            problems.append({"id": r["id"], "reason": "EXIF-rotated photo: boxes were clipped to the stored "
                                                      f"{h}x{w} at import, not the displayed {w}x{h}"})
        lines, outside = [], 0
        for b in r.get("boxes") or []:
            key = space.map(b["label"])
            if key is None:
                continue
            x0, y0, x1, y1 = b["bbox"]
            if x1 > w + 1 or y1 > h + 1:
                outside += 1
            line = yolo_line(class_index(space, args.class_mode, key), b["bbox"], w, h)
            if line:
                lines.append(line)
                counts[split][classes[int(line.split()[0])]["key"]] += 1
        if outside:
            problems.append({"id": r["id"], "reason": f"{outside} box(es) outside the {w}x{h} displayed image "
                                                      "(check the EXIF orientation of this photo)"})
        if not lines and not args.keep_empty:
            problems.append({"id": r["id"], "reason": "no usable boxes"})
            continue
        name = safe_name(r["id"]) + src.suffix.lower()
        methods[place(src, out_dir / "images" / split / name, args.link)] += 1
        (out_dir / "labels" / split / name).with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        images[split] += 1

    data_yaml = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                 "test": "images/test", "names": {i: c["key"] for i, c in enumerate(classes)}}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (out_dir / "classes.json").write_text(json.dumps(
        {"class_mode": args.class_mode, "modality": args.modality, "classes": classes}, indent=2))
    summary = {"manifest": str(args.manifest), "class_mode": args.class_mode,
               "images": dict(images), "boxes": {s: dict(c) for s, c in counts.items()},
               "placed_by": dict(methods), "problems": problems}
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))

    log.info("YOLO dataset in %s (%s)", out_dir, ", ".join(f"{m}: {n}" for m, n in methods.items()))
    for split in SPLITS:
        log.info("  %-5s %5d images  boxes: %s", split, images[split],
                 ", ".join(f"{k}={n}" for k, n in sorted(counts[split].items())) or "none")
    if problems:
        by = Counter("boxes outside image" if "outside" in p["reason"]
                     else "EXIF-rotated photo" if p["reason"].startswith("EXIF") else p["reason"]
                     for p in problems)
        log.warning("%d images with problems (details in export_summary.json): %s", len(problems),
                    ", ".join(f"{k}: {n}" for k, n in by.items()))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--modality", default="smartphone")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--class_mode", choices=CLASS_MODES, default="type",
                    help="type: one YOLO class per material class; binary: asbestos / non-asbestos; "
                         "object: a single 'material' class (where, not what).")
    ap.add_argument("--link", choices=("symlink", "hardlink", "copy"), default="symlink",
                    help="How images are placed in out_dir (a refused link falls back to a copy).")
    ap.add_argument("--include_augmented", action="store_true",
                    help="Also export offline-augmented copies (train split only).")
    ap.add_argument("--keep_empty", action="store_true",
                    help="Keep images without boxes as background images (empty label files).")
    add_label_args(ap)
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export(build_parser().parse_args())


if __name__ == "__main__":
    main()
