"""Train a YOLO detector on a dataset from src/yolo_data.py and score it on the test split.

    # once, on the login node (compute nodes have no internet):
    python -m src.yolo_train --setup --weights_dir $PROJECT_DIR/weights --model yolo11m.pt

    python -m src.yolo_train --data $D/yolo_macro/data.yaml \
        --model $PROJECT_DIR/weights/yolo11m.pt --run_dir $RUNS/yolo_macro

Writes into run_dir (Ultralytics' own output: weights/best.pt, results.csv, plots) and adds
    classes.json          copied from the dataset, read by src/autolabel.py
    test_metrics.json     mAP50, mAP50-95, precision, recall on the test split, overall and per class
    test_per_class.csv    the same per class, with the number of test images and boxes

Re-running with the same run_dir resumes an interrupted run (weights/last.pt).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import yaml

log = logging.getLogger("yolo_train")


def setup(args) -> None:
    """Fetch everything Ultralytics would download during a job: the starting weights, the
    small model its mixed-precision check loads, and the plotting font. Run on a node with
    internet; YOLO_CONFIG_DIR must point at the same folder in the job (slurm/train_yolo.sh)."""
    from ultralytics import YOLO, settings
    from ultralytics.utils.checks import check_font

    wdir = Path(args.weights_dir).resolve()
    wdir.mkdir(parents=True, exist_ok=True)
    settings.update({"weights_dir": str(wdir), "sync": False})
    for name in dict.fromkeys([Path(args.model).name, "yolo11n.pt"]):
        target = wdir / name
        if not target.exists():
            YOLO(name)                                    # downloads into the working directory
            if Path(name).exists():
                shutil.move(name, target)
        log.info("weights: %s (%s)", target, "ok" if target.exists() else "MISSING")
    check_font("Arial.ttf")
    check_font("Arial.Unicode.ttf")
    log.info("Ultralytics settings in %s; weights in %s", os.environ.get("YOLO_CONFIG_DIR", "~/.config"), wdir)


def per_class(metrics, names: dict[int, str], data_yaml: Path, split: str) -> list[dict]:
    """Per-class rows from an Ultralytics DetMetrics, with image and box counts from the labels."""
    cfg = yaml.safe_load(data_yaml.read_text())
    labels_dir = Path(cfg["path"]) / str(cfg[split]).replace("images", "labels", 1)
    n_img, n_box = {i: 0 for i in names}, {i: 0 for i in names}
    for txt in labels_dir.glob("*.txt"):
        seen = set()
        for line in txt.read_text().split("\n"):
            if line.strip():
                c = int(line.split()[0])
                n_box[c] += 1
                seen.add(c)
        for c in seen:
            n_img[c] += 1
    box = metrics.box
    rows = []
    for j, c in enumerate(metrics.ap_class_index):
        c = int(c)
        rows.append({"class": names[c], "images": n_img[c], "boxes": n_box[c],
                     "precision": float(box.p[j]), "recall": float(box.r[j]),
                     "mAP50": float(box.ap50[j]), "mAP50_95": float(box.ap[j])})
    for c in names:                                       # classes absent from the split
        if c not in {int(i) for i in metrics.ap_class_index}:
            rows.append({"class": names[c], "images": n_img[c], "boxes": n_box[c],
                         "precision": None, "recall": None, "mAP50": None, "mAP50_95": None})
    return rows


def train(args) -> None:
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    data_yaml = Path(args.data).resolve()
    run_dir = Path(args.run_dir).resolve()
    last = run_dir / "weights" / "last.pt"
    if last.exists() and not args.restart:
        log.info("Resuming %s", last)
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    patience=args.patience, workers=args.workers, seed=args.seed, device=args.device,
                    project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
                    cos_lr=True, close_mosaic=10, plots=not args.no_plots,
                    degrees=args.degrees, flipud=0.5, fliplr=0.5)

    best = run_dir / "weights" / "best.pt"
    classes_json = data_yaml.parent / "classes.json"
    if classes_json.exists():
        shutil.copy2(classes_json, run_dir / "classes.json")

    model = YOLO(str(best))
    m = model.val(data=str(data_yaml), split=args.eval_split, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, project=str(run_dir), name=f"val_{args.eval_split}", exist_ok=True,
                  plots=not args.no_plots)
    rows = per_class(m, model.names, data_yaml, args.eval_split)
    summary = {"weights": str(best), "split": args.eval_split, "imgsz": args.imgsz,
               "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
               "precision": float(m.box.mp), "recall": float(m.box.mr), "per_class": rows}
    (run_dir / f"{args.eval_split}_metrics.json").write_text(json.dumps(summary, indent=2))
    import pandas as pd
    pd.DataFrame(rows).to_csv(run_dir / f"{args.eval_split}_per_class.csv", index=False)
    log.info("%s split: mAP50 %.3f, mAP50-95 %.3f, precision %.3f, recall %.3f", args.eval_split,
             summary["mAP50"], summary["mAP50_95"], summary["precision"], summary["recall"])
    for r in rows:
        if r["mAP50"] is not None:
            log.info("  %-14s images %4d boxes %4d  P %.3f R %.3f mAP50 %.3f mAP50-95 %.3f", r["class"],
                     r["images"], r["boxes"], r["precision"], r["recall"], r["mAP50"], r["mAP50_95"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", action="store_true", help="Download weights and fonts, then exit.")
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--data", help="data.yaml written by src.yolo_data")
    ap.add_argument("--model", default="yolo11m.pt", help="Starting weights (.pt) or a model .yaml.")
    ap.add_argument("--run_dir", help="Output folder for this run.")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40, help="Stop after this many epochs without improvement.")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu; default: Ultralytics picks.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degrees", type=float, default=10.0, help="Random rotation range during training.")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--restart", action="store_true", help="Start again even if run_dir has weights/last.pt.")
    ap.add_argument("--no_plots", action="store_true")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.setup:
        setup(args)
        return
    if not args.data or not args.run_dir:
        raise SystemExit("--data and --run_dir are required (or use --setup)")
    train(args)


if __name__ == "__main__":
    main()
