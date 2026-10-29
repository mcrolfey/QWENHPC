"""Importers for datasets that are not in the <modality>/<label>/ folder layout.

* ``collect_yolo_flat``: one folder of images plus one folder of YOLO ``.txt``
  label files (same file stem). Class names come from a ``classes.txt`` /
  ``obj.names`` / ``data.yaml`` / ``notes.json`` file, found automatically
  next to the labels unless given explicitly.
* ``collect_json_folders``: a folder with ``asbestos`` / ``non-asbestos``
  sub-folders (the folder gives the binary label), each holding images and
  JSON box annotations. Supported JSON formats are detected automatically:
  LabelMe (one file per image), COCO and Label Studio exports (one file for
  many images).

Images stay where they are; rows carry absolute paths. Each row has the
image's ``boxes``; the image-level class is derived from them later
(:func:`image_label_from_boxes`).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from src.labels import normalise

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
NAME_FILES = ("classes.txt", "obj.names", "names.txt", "data.yaml", "dataset.yaml", "notes.json")
log = logging.getLogger("importers")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


# ----------------------------------------------------------------- YOLO flat
def load_class_names(path: Path) -> list[str]:
    """Class-id -> name list from classes.txt / obj.names / data.yaml / notes.json."""
    if path.suffix in (".yaml", ".yml"):
        names = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
    if path.suffix == ".json":  # Label Studio YOLO export
        cats = json.loads(path.read_text(encoding="utf-8"))["categories"]
        return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def find_class_names(*folders: Path) -> Path | None:
    for folder in folders:
        for name in NAME_FILES:
            if (folder / name).is_file():
                return folder / name
    return None


def collect_yolo_flat(images_dir: Path, labels_dir: Path, modality: str,
                      names_file: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Rows for a flat images/ + labels/ YOLO dataset; also returns problems."""
    names_file = names_file or find_class_names(labels_dir, labels_dir.parent, images_dir, images_dir.parent)
    if names_file is None:
        raise SystemExit(
            f"No class-name file found for the YOLO labels in {labels_dir}. Looked for {', '.join(NAME_FILES)} "
            "in the labels/images folders and their parent. Pass --micro_names with a text file listing the "
            "class names in class-id order (line 1 = id 0).")
    names = load_class_names(names_file)
    labels = {p.stem: p for p in labels_dir.rglob("*.txt") if p.name not in NAME_FILES}
    if not names:
        ids: Counter = Counter()
        for txt in tqdm(labels.values(), desc="scanning YOLO class ids"):
            for ln in txt.read_text(encoding="utf-8").splitlines():
                if ln.split():
                    ids[int(float(ln.split()[0]))] += 1
        raise SystemExit(
            f"The class-name file {names_file} is empty, so the YOLO class ids cannot be named.\n"
            f"Class ids used in the {len(labels)} label files (id: number of boxes): {dict(sorted(ids.items()))}\n"
            "Create a text file with one class code per line in id order (line 1 = id 0, e.g. A-AM) and "
            "pass it with --micro_names <file>.")
    log.info("YOLO class names from %s: %s", names_file, dict(enumerate(names)))
    rows, problems = [], []
    for img in tqdm(list_images(images_dir), desc="micro images"):
        txt = labels.get(img.stem)
        if txt is None:
            problems.append({"image": str(img), "modality": modality, "reason": "no label file"})
            continue
        w, h = image_size(img)
        boxes = []
        for ln in txt.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if cid >= len(names):
                problems.append({"image": str(img), "modality": modality,
                                 "reason": f"class id {cid} not in class-name file"})
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            boxes.append(_box(names[cid], (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                              (cy + bh / 2) * h, w, h))
        rows.append({"image": str(img.resolve()), "modality": modality, "width": w, "height": h,
                     "boxes": boxes, "label": None, "binary_hint": None})
    return rows, problems


# ------------------------------------------------------------- JSON folders
def _box(label: str, x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> dict:
    x0, x1 = sorted((max(0.0, x0), min(float(w), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(h), y1)))
    return {"bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)], "label": str(label)}


def _labelme(data: dict) -> list[tuple]:
    """LabelMe shapes (rectangles or polygons) as absolute boxes; clipped later."""
    boxes = []
    for s in data.get("shapes", []):
        pts = s.get("points") or []
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        boxes.append(("abs", s.get("label", ""), min(xs), min(ys), max(xs), max(ys)))
    return boxes


def _lookup_keys(name: str) -> list[str]:
    """Keys under which an annotated file name is indexed: its name and, for
    Label Studio uploads ("1a2b3c4d-photo.jpg"), the name without the prefix."""
    keys = [name]
    if re.match(r"^[0-9a-f]{8}-", name):
        keys.append(name[9:])
    return keys


def _index_coco(data: dict) -> dict[str, list[tuple]]:
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    files = {im["id"]: Path(im["file_name"]).name for im in data.get("images", [])}
    out: dict[str, list[tuple]] = {}
    for a in data.get("annotations", []):
        x, y, bw, bh = a["bbox"]
        out.setdefault(files.get(a["image_id"], ""), []).append(("abs", cats.get(a["category_id"], ""),
                                                                  x, y, x + bw, y + bh))
    for fname in files.values():
        out.setdefault(fname, [])
    return out


def _index_label_studio(data: list) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for task in data:
        src = next(iter((task.get("data") or {}).values()), "")
        fname = Path(str(src).split("?")[0]).name
        results = []
        for ann in task.get("annotations") or task.get("completions") or []:
            results += ann.get("result", [])
        boxes = out.setdefault(fname, [])
        for r in results:
            v = r.get("value", {})
            labels = v.get("rectanglelabels") or v.get("labels") or []
            if not labels or "width" not in v:
                continue
            boxes.append(("pct", labels[0], v["x"], v["y"], v["x"] + v["width"], v["y"] + v["height"]))
    return out


def detect_format(data) -> str:
    if isinstance(data, dict) and "shapes" in data:
        return "labelme"
    if isinstance(data, dict) and "annotations" in data and "images" in data:
        return "coco"
    if isinstance(data, list) and data and isinstance(data[0], dict) and ("annotations" in data[0] or "completions" in data[0]):
        return "label_studio"
    keys = list(data)[:10] if isinstance(data, dict) else f"list of {type(data[0]).__name__}" if data else "empty"
    return f"unknown (top-level: {keys})"


def collect_json_folders(root: Path, modality: str) -> tuple[list[dict], list[dict], Counter]:
    """Rows for <root>/<asbestos|non-asbestos>/ folders with images + JSON boxes."""
    rows, problems, formats = [], [], Counter()
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not subdirs:
        raise SystemExit(f"No sub-folders in {root}; expected e.g. asbestos/ and non-asbestos/")
    for sub in subdirs:
        n = normalise(sub.name)
        hint = False if n.startswith(("non", "na_", "no_")) else True if "asbestos" in n else None
        if hint is None:
            log.warning("Folder %s is neither 'asbestos' nor 'non-asbestos'; no binary check for it", sub)
        per_image: dict[str, list[tuple]] = {}   # LabelMe: json stem -> boxes
        shared: dict[str, list[tuple]] = {}      # COCO / Label Studio: file name -> boxes
        for js in tqdm(sorted(sub.rglob("*.json")), desc=f"{sub.name} JSON"):
            try:
                data = json.loads(js.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                problems.append({"image": str(js), "modality": modality, "reason": f"unreadable JSON: {exc}"})
                continue
            fmt = detect_format(data)
            formats[fmt] += 1
            if fmt == "labelme":
                per_image[js.stem] = _labelme(data)
            elif fmt in ("coco", "label_studio"):
                index = _index_coco(data) if fmt == "coco" else _index_label_studio(data)
                for name, boxes in index.items():
                    for k in _lookup_keys(name):
                        shared.setdefault(k, []).extend(boxes)
            else:
                problems.append({"image": str(js), "modality": modality, "reason": f"JSON format {fmt}"})
            del data
        for img in tqdm(list_images(sub), desc=f"{sub.name} images"):
            raw = per_image.get(img.stem)
            if raw is None:
                raw = shared.get(img.name)
            if raw is None:
                problems.append({"image": str(img), "modality": modality, "reason": "no JSON annotation"})
                continue
            w, h = image_size(img)
            boxes = []
            for kind, lab, x0, y0, x1, y1 in raw:
                if kind == "pct":
                    x0, x1, y0, y1 = x0 * w / 100, x1 * w / 100, y0 * h / 100, y1 * h / 100
                boxes.append(_box(lab, x0, y0, x1, y1, w, h))
            rows.append({"image": str(img.resolve()), "modality": modality, "width": w, "height": h,
                         "boxes": boxes, "label": None, "binary_hint": hint, "folder": sub.name})
    return rows, problems, formats


# --------------------------------------------------------- image-level label
def image_label_from_boxes(boxes: list[dict], space) -> tuple[str | None, str]:
    """Image class from its boxes: the asbestos class with the largest total box
    area if any asbestos box exists, else the largest non-asbestos class.

    Returns (raw label or None, note). Boxes whose label the space drops are ignored.
    """
    area: Counter = Counter()
    raw_of: dict[str, str] = {}
    for b in boxes:
        key = space.map(b["label"])
        if key is None:
            continue
        x0, y0, x1, y1 = b["bbox"]
        area[key] += max(0.0, x1 - x0) * max(0.0, y1 - y0) or 1e-6
        raw_of.setdefault(key, b["label"])
    if not area:
        return None, "no usable boxes"
    asb = {k: a for k, a in area.items() if space.is_asbestos(k)}
    pool = asb or area
    key = max(pool, key=pool.get)
    note = "multiple classes" if len(area) > 1 else ""
    return raw_of[key], note
