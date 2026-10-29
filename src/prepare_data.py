"""Build a manifest (manifest.jsonl) from raw images, with leakage-free splits.

Two input modes:

1. Folder layout (default)::

       <data_dir>/<raw_subdir>/<modality>/<label>/<image files>
       e.g. data/raw/microscopic/A-CF/S012_03.tif

2. CSV (``--csv``) with columns ``image,label,modality`` and optional
   ``sample_id,split``. ``image`` is relative to ``data_dir``.

3. Existing folders, read in place (``--micro_images/--micro_labels`` for a
   flat YOLO dataset, ``--macro_dir`` for asbestos/ + non-asbestos/ folders
   with JSON boxes; see src/importers.py). The image class is derived from its
   boxes (largest asbestos class, else largest non-asbestos class). Images
   without annotations, or whose boxes contradict their asbestos/non-asbestos
   folder, are skipped and listed in ``<data_dir>/prepare_report.csv``.
   ``--dry_run`` reports labels, formats and counts without writing.

Bounding boxes are read from a YOLO ``.txt`` file with the same stem next to
each image (``class cx cy w h`` normalised), with class ids resolved through
``--yolo_names`` (a names.txt with one class per line, or a YOLO data.yaml).

Splits are grouped by ``sample_id`` *across modalities*, so every image of a
physical specimen (microscope and phone) lands in the same split. This
prevents leakage and makes paired McNemar tests possible. ``sample_id`` is
extracted from the file stem with ``--sample_id_regex`` (first group).

Example::

    python -m src.prepare_data --data_dir /scratch/$USER/asbestos/data \
        --sample_id_regex '^(S\\d+)' --val_frac 0.15 --test_frac 0.15
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src import MODALITIES
from src.importers import collect_json_folders, collect_yolo_flat, image_label_from_boxes
from src.labels import add_label_args, label_space_from_args, normalise
from src.manifest import summarise, write_manifest

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
log = logging.getLogger("prepare_data")


def load_yolo_names(path: str | None) -> list[str] | None:
    if not path:
        return None
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        names = cfg["names"]
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_yolo_boxes(img_path: Path, width: int, height: int, names: list[str] | None, default_label: str) -> list[dict]:
    txt = img_path.with_suffix(".txt")
    if not txt.exists():
        return []
    boxes = []
    for ln in txt.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        label = names[cls_id] if names and cls_id < len(names) else default_label
        x0, y0 = (cx - w / 2) * width, (cy - h / 2) * height
        x1, y1 = (cx + w / 2) * width, (cy + h / 2) * height
        boxes.append({
            "bbox": [round(max(0.0, x0), 2), round(max(0.0, y0), 2),
                     round(min(width, x1), 2), round(min(height, y1), 2)],
            "label": label,
        })
    return boxes


def _maps(space, label: str) -> bool:
    try:
        space.map(label)
        return True
    except ValueError:
        return False


def sample_id_for(stem: str, regex: str | None) -> str:
    if regex:
        m = re.search(regex, stem)
        if m:
            return m.group(1) if m.groups() else m.group(0)
        log.warning("sample_id_regex did not match '%s'; using full stem", stem)
    return stem


def collect_from_folders(data_dir: Path, raw_subdir: str) -> list[dict]:
    root = data_dir / raw_subdir
    rows = []
    for modality in MODALITIES:
        mod_dir = root / modality
        if not mod_dir.is_dir():
            log.warning("Missing modality folder %s", mod_dir)
            continue
        for label_dir in sorted(p for p in mod_dir.iterdir() if p.is_dir()):
            for img in sorted(label_dir.rglob("*")):
                if img.suffix.lower() in IMAGE_EXTS:
                    rows.append({"image": img.relative_to(data_dir).as_posix(),
                                 "label": label_dir.name, "modality": modality})
    return rows


def collect_from_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for r in rows:
        missing = {"image", "label", "modality"} - set(r)
        if missing:
            raise ValueError(f"CSV is missing columns {missing}")
        if r["modality"] not in MODALITIES:
            raise ValueError(f"Bad modality '{r['modality']}' (expected one of {MODALITIES})")
    return rows


def grouped_split(records: list[dict], val_frac: float, test_frac: float, seed: int) -> None:
    """Assign train/val/test per sample_id, stratified by the sample's class.

    Uses a greedy per-class allocation of whole groups, which stays exact on
    small datasets where k-fold based splitters break down.
    """
    group_label: dict[str, str] = {}
    group_size: Counter = Counter()
    for r in records:
        sid = r["sample_id"]
        group_size[sid] += 1
        if sid in group_label and group_label[sid] != r["_key"]:
            log.warning("sample_id %s has mixed labels (%s, %s); stratifying on the first",
                        sid, group_label[sid], r["_key"])
        group_label.setdefault(sid, r["_key"])

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for sid, lab in group_label.items():
        by_class[lab].append(sid)

    assignment: dict[str, str] = {}
    for lab, sids in sorted(by_class.items()):
        sids = sorted(sids)
        rng.shuffle(sids)
        total = sum(group_size[s] for s in sids)
        targets = {"test": test_frac * total, "val": val_frac * total}
        filled = {"test": 0, "val": 0}
        for sid in sids:  # fill test, then val, remainder is train
            split = next((n for n in ("test", "val") if filled[n] < targets[n]), "train")
            assignment[sid] = split
            if split != "train":
                filled[split] += group_size[sid]
        # Always keep at least one group of every class for training.
        if not any(assignment[s] == "train" for s in sids):
            for name in ("val", "test"):
                victim = next((s for s in reversed(sids) if assignment[s] == name), None)
                if victim is not None:
                    assignment[victim] = "train"
                    break
        counts = Counter(assignment[s] for s in sids)
        if len(sids) < 3:
            log.warning("Class '%s' has only %d sample groups; splits: %s", lab, len(sids), dict(counts))
    for r in records:
        r["split"] = assignment[r["sample_id"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--raw_subdir", default="raw", help="Folder mode: images under <data_dir>/<raw_subdir>.")
    ap.add_argument("--csv", default=None, help="CSV input instead of the folder layout.")
    ap.add_argument("--output", default=None, help="Default: <data_dir>/manifest.jsonl")
    ap.add_argument("--yolo_names", default=None, help="names.txt or data.yaml for YOLO class ids.")
    ap.add_argument("--sample_id_regex", default=None,
                    help=r"Regex on file stem; group 1 is the specimen id (e.g. '^(S\d+)').")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--keep_splits", action="store_true", help="Keep 'split' column from the CSV.")
    g = ap.add_argument_group("import existing folders in place (instead of the raw/ layout)")
    g.add_argument("--micro_images", default=None, help="Folder of microscopic images (flat).")
    g.add_argument("--micro_labels", default=None, help="Folder of YOLO .txt labels for --micro_images.")
    g.add_argument("--micro_names", default=None,
                   help="Class-name file for the YOLO ids (default: classes.txt/obj.names/data.yaml/notes.json "
                        "found next to the labels).")
    g.add_argument("--macro_dir", default=None,
                   help="Folder with asbestos/ and non-asbestos/ sub-folders of macro (smartphone) images "
                        "and JSON box files (LabelMe, COCO or Label Studio).")
    g.add_argument("--dry_run", action="store_true", help="Report what was found; write no manifest.")
    ap.add_argument("--seed", type=int, default=42)
    add_label_args(ap)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_dir = Path(args.data_dir)
    space = label_space_from_args(args)
    log.info("\n%s", space.describe())
    data_dir.mkdir(parents=True, exist_ok=True)
    problems: list[dict] = []
    if args.micro_images or args.macro_dir:
        rows = []
        if args.micro_images:
            if not args.micro_labels:
                raise SystemExit("--micro_labels is required with --micro_images")
            r, p = collect_yolo_flat(Path(args.micro_images), Path(args.micro_labels), "microscopic",
                                     Path(args.micro_names) if args.micro_names else None)
            rows += r
            problems += p
        if args.macro_dir:
            r, p, formats = collect_json_folders(Path(args.macro_dir), "smartphone")
            log.info("Macro JSON files by format: %s", dict(formats))
            rows += r
            problems += p
    else:
        names = load_yolo_names(args.yolo_names)
        rows = collect_from_csv(Path(args.csv)) if args.csv else collect_from_folders(data_dir, args.raw_subdir)
        for r in rows:
            path = Path(r["image"])
            abs_path = path if path.is_absolute() else data_dir / path
            with Image.open(abs_path) as im:
                r["width"], r["height"] = im.size
            r["boxes"] = read_yolo_boxes(abs_path, r["width"], r["height"], names, r["label"])
    if not rows:
        raise SystemExit("No images found.")

    # Every raw label (image and box) must be a class, alias or dropped label.
    raw_counts: Counter = Counter()
    for r in rows:
        for lab in ([r["label"]] if r.get("label") else []) + [b["label"] for b in r.get("boxes") or []]:
            raw_counts[(r["modality"], lab)] += 1
    unknown = sorted({lab for (_, lab) in raw_counts if not _maps(space, lab)})
    lines = []
    for (mod, lab), n in sorted(raw_counts.items()):
        tgt = "UNKNOWN" if lab in unknown else space.map(lab)
        lines.append(f"  {mod:<12} {lab!s:<28} -> {tgt}  ({n} labels)")
    log.info("Raw labels found (image + box labels):\n%s", "\n".join(lines))
    if unknown:
        msg = (f"Labels not in the class config: {unknown}. Add them under 'aliases' (e.g. "
               f"'{normalise(unknown[0])}: a_cry') or 'drop' in configs/classes_*.yaml.")
        if not args.dry_run:
            raise SystemExit(msg)
        log.warning(msg)

    records, dropped, notes, seen_ids = [], Counter(), Counter(), Counter()
    for r in rows:
        if unknown and any(not _maps(space, b["label"]) for b in r.get("boxes") or []):
            continue  # dry run only
        if r.get("label"):
            label, note = r["label"], ""
        else:
            label, note = image_label_from_boxes(r.get("boxes") or [], space)
        if label is None:
            problems.append({"image": r["image"], "modality": r["modality"], "reason": note})
            continue
        key = space.map(label)
        if key is None:
            dropped[label] += 1
            continue
        hint = r.get("binary_hint")
        if hint is not None and space.is_asbestos(key) != hint:
            problems.append({"image": r["image"], "modality": r["modality"],
                             "reason": f"folder '{r.get('folder')}' disagrees with box class {label}"})
            continue
        if note:
            notes[note] += 1
        path = Path(r["image"])
        abs_path = path if path.is_absolute() else data_dir / path
        rid = f"{r['modality']}/{path.stem}" if path.is_absolute() else path.with_suffix("").as_posix()
        seen_ids[rid] += 1
        if seen_ids[rid] > 1:
            rid = f"{rid}__{seen_ids[rid]}"
        rec = {
            "id": rid,
            "image": r["image"],
            "label": label,
            "modality": r["modality"],
            "sample_id": r.get("sample_id") or sample_id_for(abs_path.stem, args.sample_id_regex),
            "width": int(r["width"]),
            "height": int(r["height"]),
            "boxes": r.get("boxes") or [],
            "_key": key,
        }
        if args.keep_splits and r.get("split"):
            rec["split"] = r["split"]
        records.append(rec)

    if problems:
        report = data_dir / "prepare_report.csv"
        with open(report, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["image", "modality", "reason"])
            w.writeheader()
            w.writerows(problems)
        by_reason = Counter((p["modality"], p["reason"].split(":")[0]) for p in problems)
        log.warning("%d images skipped (details in %s):\n%s", len(problems), report,
                    "\n".join(f"  {m:<12} {r}: {n}" for (m, r), n in sorted(by_reason.items())))
    if notes:
        log.info("Images with boxes of several classes (labelled by the largest asbestos class): %d",
                 notes["multiple classes"])
    if dropped:
        log.info("Dropped images by label: %s", dict(dropped))
    if args.dry_run:
        log.info("Dry run: %d usable images; per modality and class:\n%s", len(records),
                 "\n".join(f"  {m:<12} {k:<14} {n}" for (m, k), n in
                            sorted(Counter((r["modality"], r["_key"]) for r in records).items())))
        return
    if not records:
        raise SystemExit("No usable images.")
    if not args.keep_splits or any("split" not in r for r in records):
        grouped_split(records, args.val_frac, args.test_frac, args.seed)
    for r in records:
        r.pop("_key")

    out = Path(args.output) if args.output else data_dir / "manifest.jsonl"
    write_manifest(records, out)
    paired = defaultdict(set)
    for r in records:
        paired[r["sample_id"]].add(r["modality"])
    n_paired = sum(1 for m in paired.values() if len(m) == len(MODALITIES))
    log.info("Wrote %d records to %s", len(records), out)
    log.info("Specimens with both modalities (usable for paired tests): %d / %d", n_paired, len(paired))
    log.info("Counts (raw labels):\n%s", summarise(records))


if __name__ == "__main__":
    main()
