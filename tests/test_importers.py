"""Import of the flat YOLO (micro) and JSON-folder (macro) datasets."""

import json
import subprocess
import sys

import numpy as np
from PIL import Image

from conftest import ROOT
from src.manifest import read_manifest


def _img(path, w=200, h=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(path)


def _make(tmp_path):
    combined = tmp_path / "combinedImgs"
    # micro: flat images/ + labels/ (YOLO), class names next to them
    (combined / "labels").mkdir(parents=True)
    (combined / "classes.txt").write_text("A-AM\nA-CF\nNA-CS\n")
    _img(combined / "images" / "m1.png")
    (combined / "labels" / "m1.txt").write_text("1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.05 0.05\n")  # A-CF larger
    _img(combined / "images" / "m2.png")
    (combined / "labels" / "m2.txt").write_text("2 0.5 0.5 0.5 0.5\n")
    _img(combined / "images" / "m3.png")                                                   # no label file
    # macro: asbestos/ with LabelMe, non-asbestos/ with COCO
    asb = combined / "macro_asbestos" / "asbestos"
    _img(asb / "p1.jpg")
    (asb / "p1.json").write_text(json.dumps({"shapes": [
        {"label": "A-CRY", "points": [[10, 10], [60, 40]], "shape_type": "rectangle"}]}))
    _img(asb / "p2.jpg")
    (asb / "p2.json").write_text(json.dumps({"shapes": [
        {"label": "NA-CS", "points": [[10, 10], [60, 40]], "shape_type": "rectangle"}]}))  # conflicts with folder
    _img(asb / "p3.jpg")                                                                    # no JSON
    non = combined / "macro_asbestos" / "non-asbestos"
    _img(non / "n1.jpg")
    (non / "coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "n1.jpg"}],
        "categories": [{"id": 4, "name": "NA-CS"}],
        "annotations": [{"image_id": 1, "category_id": 4, "bbox": [5, 5, 50, 30]}]}))
    return combined


def _run(tmp_path, combined, *extra):
    cmd = [sys.executable, "-m", "src.prepare_data", "--data_dir", str(tmp_path / "data"),
           "--micro_images", str(combined / "images"), "--micro_labels", str(combined / "labels"),
           "--macro_dir", str(combined / "macro_asbestos"), "--val_frac", "0", "--test_frac", "0", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def test_import_flat_yolo_and_json_folders(tmp_path):
    combined = _make(tmp_path)
    res = _run(tmp_path, combined)
    assert res.returncode == 0, res.stderr
    recs = {r["id"]: r for r in read_manifest(tmp_path / "data" / "manifest.jsonl")}
    assert set(recs) == {"microscopic/m1", "microscopic/m2", "smartphone/p1", "smartphone/n1"}
    assert recs["microscopic/m1"]["label"] == "A-CF"          # largest asbestos box wins
    assert recs["microscopic/m1"]["boxes"][0]["bbox"] == [80.0, 40.0, 120.0, 60.0]
    assert recs["smartphone/p1"]["label"] == "A-CRY"
    assert recs["smartphone/n1"]["boxes"][0]["bbox"] == [5.0, 5.0, 55.0, 35.0]   # COCO xywh -> xyxy
    report = (tmp_path / "data" / "prepare_report.csv").read_text()
    assert "m3.png" in report and "p3.jpg" in report and "disagrees" in report


def test_unknown_labels_are_reported(tmp_path):
    combined = _make(tmp_path)
    (combined / "classes.txt").write_text("A-AM\nCHRYSO\nNA-CS\n")
    res = _run(tmp_path, combined)
    assert res.returncode != 0 and "CHRYSO" in res.stderr
    dry = _run(tmp_path, combined, "--dry_run")
    assert dry.returncode == 0 and "UNKNOWN" in dry.stderr
    assert not (tmp_path / "data" / "manifest.jsonl").exists()


def test_empty_class_name_file_stops_with_ids(tmp_path):
    combined = _make(tmp_path)
    (combined / "classes.txt").write_text("")
    res = _run(tmp_path, combined, "--dry_run")
    assert res.returncode != 0
    assert "is empty" in res.stderr and "{0: 1, 1: 1, 2: 1}" in res.stderr and "--micro_names" in res.stderr


def test_class_names_utf16_and_empty_file_skipped(tmp_path):
    combined = _make(tmp_path)
    (combined / "labels" / "classes.txt").write_text("")                                  # empty, found first
    (combined / "classes.txt").write_text("A-AM\r\nA-CF\r\nNA-CS\r\n", encoding="utf-16")  # Windows tool output
    res = _run(tmp_path, combined, "--dry_run")
    assert res.returncode == 0, res.stderr
    assert "Ignoring empty" in res.stderr and "{0: 'A-AM', 1: 'A-CF', 2: 'NA-CS'}" in res.stderr
