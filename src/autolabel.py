"""Pre-label images with a trained YOLO model, as LabelMe JSON files to review and correct.

    # new, unlabelled photos (one JSON next to each image, as LabelMe expects):
    python -m src.autolabel --weights $RUNS/yolo_macro/weights/best.pt --images /path/to/new_photos

    # check the labeller on images that already have human boxes (nothing is overwritten:
    # JSON files go to --out_dir, and image classes are compared with the manifest's):
    python -m src.autolabel --weights $RUNS/yolo_macro/weights/best.pt \
        --manifest $D/manifest_grouped.jsonl --data_dir $D --split test --out_dir $R/autolabel_test

Each box becomes a LabelMe rectangle whose label is the raw class name used in the existing
annotations (from classes.json), so the files import with ``prepare_data --macro_dir`` like
hand-made ones. The shape's description holds the confidence ("auto 0.87"), and the file
carries an "autolabel" entry (model, thresholds) that LabelMe keeps when the file is saved.

An existing JSON is never replaced unless --overwrite is given, so finished human labels are safe.
review.csv lists every image with its box count, lowest confidence and implied image class,
least certain first: the order to check them in.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.importers import image_label_from_boxes, list_images
from src.labels import add_label_args, label_space_from_args
from src.manifest import read_manifest, resolve_image

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger("autolabel")
LABELME_VERSION = "5.4.1"


def load_displayed(path: Path) -> np.ndarray:
    """Image as LabelMe shows it (EXIF orientation applied), BGR for Ultralytics."""
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def labelme_doc(image_path: Path, json_path: Path, width: int, height: int, shapes: list[dict],
                meta: dict) -> dict:
    try:
        rel = Path(image_path.resolve()).relative_to(json_path.parent.resolve()).as_posix()
    except ValueError:
        rel = str(image_path.resolve())
    return {"version": LABELME_VERSION, "flags": {}, "shapes": shapes, "imagePath": rel,
            "imageData": None, "imageHeight": height, "imageWidth": width, "autolabel": meta}


def to_shapes(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, classes: list[dict]) -> list[dict]:
    shapes = []
    for (x0, y0, x1, y1), c, k in zip(xyxy.tolist(), conf.tolist(), cls.tolist()):
        shapes.append({"label": classes[int(k)]["raw_label"],
                       "points": [[round(x0, 2), round(y0, 2)], [round(x1, 2), round(y1, 2)]],
                       "group_id": None, "description": f"auto {c:.2f}", "shape_type": "rectangle",
                       "flags": {}, "mask": None})
    return shapes


def image_class(shapes: list[dict], space) -> str | None:
    """Image class implied by the boxes, by the same rule used to label the training images."""
    boxes = [{"label": s["label"], "bbox": [*s["points"][0], *s["points"][1]]} for s in shapes]
    raw, _ = image_label_from_boxes(boxes, space)
    return space.map(raw) if raw else None


def sources(args) -> list[tuple[Path, Path, dict | None]]:
    """(image, json path, manifest record or None) for every image to label."""
    out_dir = Path(args.out_dir) if args.out_dir else None
    items = []
    if args.images:
        root = Path(args.images)
        for img in list_images(root):
            js = (out_dir / img.relative_to(root)).with_suffix(".json") if out_dir else img.with_suffix(".json")
            items.append((img, js, None))
    else:
        if not out_dir:
            raise SystemExit("--manifest needs --out_dir, so the existing human labels stay untouched")
        recs = [r for r in read_manifest(args.manifest)
                if r.get("modality") == args.modality and not r.get("augmented_from")
                and (not args.split or r.get("split") in args.split)]
        for r in recs:
            img = resolve_image(r, args.data_dir)
            items.append((img, out_dir / (r["id"].replace("/", "__") + ".json"), r))
    return items


def run(args) -> pd.DataFrame:
    from ultralytics import YOLO

    weights = Path(args.weights)
    cj = Path(args.classes_json) if args.classes_json else weights.parent.parent / "classes.json"
    if not cj.exists():
        raise SystemExit(f"classes.json not found at {cj}; pass --classes_json (written by src.yolo_data)")
    cinfo = json.loads(cj.read_text())
    classes, mode = cinfo["classes"], cinfo["class_mode"]
    full_space = label_space_from_args(args, modality=args.modality)
    space = full_space if mode == "type" else None

    model = YOLO(str(weights))
    if len(model.names) != len(classes):
        raise SystemExit(f"{weights} has {len(model.names)} classes but {cj} lists {len(classes)}")
    meta = {"model": str(weights.resolve()), "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz}

    rows, skipped = [], 0
    for img, js, rec in sources(args):
        if js.exists() and not args.overwrite:
            skipped += 1
            continue
        arr = load_displayed(img)
        h, w = arr.shape[:2]
        res = model.predict(arr, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
                            max_det=args.max_det, agnostic_nms=args.agnostic_nms, verbose=False)[0]
        b = res.boxes
        shapes = to_shapes(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), classes)
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(labelme_doc(img, js, w, h, shapes, meta), indent=2))

        conf = b.conf.cpu().numpy()
        row = {"image": str(img), "json": str(js), "boxes": len(shapes),
               "min_conf": float(conf.min()) if len(conf) else None,
               "mean_conf": float(conf.mean()) if len(conf) else None,
               "classes": ";".join(sorted({s["label"] for s in shapes}))}
        if mode == "type":
            row["pred_class"] = image_class(shapes, space)
        elif mode == "binary":
            row["pred_asbestos"] = (any(classes[int(k)]["asbestos"] for k in b.cls.cpu().numpy())
                                    if len(shapes) else None)
        if rec is not None and space is not None:
            row["true_class"] = space.map(rec["label"])
        elif rec is not None and mode == "binary":
            row["true_asbestos"] = full_space.is_asbestos(full_space.map(rec["label"]))
        rows.append(row)

    df = pd.DataFrame(rows)
    report_dir = Path(args.out_dir) if args.out_dir else Path(args.images)
    if df.empty:
        log.info("Nothing labelled (%d images already had a JSON; --overwrite replaces them)", skipped)
        return df
    # Least certain first: images with no box, then by their lowest box confidence.
    df = (df.assign(_found=df["boxes"] > 0).sort_values(["_found", "min_conf"])
            .drop(columns="_found").reset_index(drop=True))
    df.to_csv(report_dir / "review.csv", index=False)
    log.info("Labelled %d images (%d skipped: JSON already there); %d with no box. Review list: %s",
             len(df), skipped, int((df["boxes"] == 0).sum()), report_dir / "review.csv")
    per_label = Counter(lab for labs in df["classes"] for lab in labs.split(";") if lab)
    log.info("Images with a box of each class: %s", dict(per_label))
    summarise_agreement(df, report_dir)
    return df


def summarise_agreement(df: pd.DataFrame, out: Path) -> None:
    """With manifest labels: how often the boxes imply the right image class."""
    if "true_class" in df:
        ok = df["pred_class"] == df["true_class"]
        per = df.assign(ok=ok).groupby("true_class")["ok"].agg(["sum", "count"])
        log.info("Image class implied by the boxes matches the label on %d / %d images (%.3f)",
                 int(ok.sum()), len(df), ok.mean())
        for k, r in per.iterrows():
            log.info("  %-14s %3d / %3d", k, r["sum"], r["count"])
        conf = pd.crosstab(df["true_class"], df["pred_class"].fillna("(no box)"))
        conf.to_csv(out / "image_class_confusion.csv")
        (out / "agreement.json").write_text(json.dumps(
            {"images": len(df), "correct": int(ok.sum()), "accuracy": float(ok.mean()),
             "per_class": {k: {"correct": int(r["sum"]), "images": int(r["count"])} for k, r in per.iterrows()}},
            indent=2))
    elif "true_asbestos" in df:
        ok = df["pred_asbestos"] == df["true_asbestos"]
        log.info("Asbestos / non-asbestos implied by the boxes matches on %d / %d images (%.3f)",
                 int(ok.sum()), len(df), ok.mean())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="best.pt from src.yolo_train")
    ap.add_argument("--classes_json", default=None, help="Default: classes.json in the run folder.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="Folder of images to label (searched recursively).")
    src.add_argument("--manifest", help="Label the manifest's images instead (needs --out_dir).")
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--modality", default="smartphone")
    ap.add_argument("--split", nargs="+", default=None, help="With --manifest: only these splits.")
    ap.add_argument("--out_dir", default=None,
                    help="Write the JSON files here (mirroring the folders) instead of next to the images.")
    ap.add_argument("--conf", type=float, default=0.25, help="Lowest confidence kept as a box.")
    ap.add_argument("--iou", type=float, default=0.5, help="Overlap above which boxes are merged (NMS).")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max_det", type=int, default=100)
    ap.add_argument("--agnostic_nms", action="store_true",
                    help="Merge overlapping boxes of different classes (one class per region).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true", help="Replace JSON files that already exist.")
    add_label_args(ap)
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
