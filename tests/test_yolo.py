"""YOLO export, training and autolabelling on a few synthetic macro photos."""

import json

import numpy as np
import pytest
from PIL import Image

from conftest import MACRO_CONFIG
from src.manifest import write_manifest
from src.yolo_data import build_parser as export_parser
from src.yolo_data import export, oriented_size, yolo_line


def _photo(path, w, h, box, orientation=None):
    a = np.full((h, w, 3), 200, np.uint8)
    x0, y0, x1, y1 = box
    a[y0:y1, x0:x1] = (40, 40, 160)
    im = Image.fromarray(a)
    if orientation:
        ex = im.getexif()
        ex[274] = orientation
        im.save(path, exif=ex)
    else:
        im.save(path)


def _dataset(tmp_path, n=4):
    """n photos per split; asbestos photos have a 'Crysotile' box, the others a cement-sheet box."""
    recs = []
    for split in ("train", "val", "test"):
        for i in range(n):
            asb = i % 2 == 0
            img = tmp_path / "raw" / ("asbestos" if asb else "non-asbestos") / f"{split}_{i}.jpg"
            img.parent.mkdir(parents=True, exist_ok=True)
            box = (8, 8, 40, 40) if asb else (24, 20, 60, 56)
            _photo(img, 64, 64, box)
            recs.append({"id": f"smartphone/{img.parent.name}/{img.stem}", "image": str(img), "split": split,
                         "modality": "smartphone", "label": "Crysotile" if asb else "NA-CS",
                         "width": 64, "height": 64,
                         "boxes": [{"bbox": list(box), "label": "Crysotile" if asb else "NA-CS"}]})
    recs.append({**recs[0], "id": recs[0]["id"] + "_aug0", "augmented_from": recs[0]["id"]})
    write_manifest(recs, tmp_path / "manifest.jsonl")
    return recs


def _export(tmp_path, *extra):
    args = export_parser().parse_args(["--data_dir", str(tmp_path), "--manifest", str(tmp_path / "manifest.jsonl"),
                                       "--out_dir", str(tmp_path / "yolo"), "--class_config", str(MACRO_CONFIG),
                                       "--link", "copy", *extra])
    return export(args)


def test_yolo_line_normalises_and_clips():
    assert yolo_line(2, [10, 20, 30, 60], 100, 200) == "2 0.200000 0.200000 0.200000 0.200000"
    assert yolo_line(0, [-5, 0, 50, 10], 100, 100) == "0 0.250000 0.050000 0.500000 0.100000"
    assert yolo_line(0, [10, 10, 10.5, 40], 100, 100) is None        # thinner than a pixel


def test_oriented_size_follows_exif(tmp_path):
    p = tmp_path / "turned.jpg"
    _photo(p, 80, 40, (0, 0, 10, 10), orientation=6)
    assert oriented_size(p) == (40, 80, True)


def test_export_layout_classes_and_splits(tmp_path):
    _dataset(tmp_path)
    s = _export(tmp_path)
    assert s["images"] == {"train": 4, "val": 4, "test": 4}            # augmented copy left out
    assert s["boxes"]["test"] == {"a_cry": 2, "na_cs": 2}
    classes = json.loads((tmp_path / "yolo" / "classes.json").read_text())["classes"]
    assert [c["key"] for c in classes] == ["a_cry", "a_cry_of", "a_cry_of_cs", "na_cs"]
    assert classes[0]["raw_label"] == "Crysotile"                        # spelling used in the data
    label = (tmp_path / "yolo" / "labels" / "test" / "smartphone__asbestos__test_0.txt").read_text()
    assert label.strip() == "0 0.375000 0.375000 0.500000 0.500000"
    assert (tmp_path / "yolo" / "images" / "test" / "smartphone__asbestos__test_0.jpg").exists()
    assert s["problems"] == []

    s = _export(tmp_path, "--class_mode", "binary")
    assert s["boxes"]["test"] == {"asbestos": 2, "non_asbestos": 2}


def test_export_flags_rotated_photos_clipped_at_import(tmp_path):
    recs = _dataset(tmp_path, n=2)
    turned = tmp_path / "raw" / "turned.jpg"
    _photo(turned, 80, 40, (0, 0, 10, 10), orientation=6)                # displayed 40 wide, 80 high
    recs.append({**recs[0], "id": "smartphone/turned", "image": str(turned), "width": 80, "height": 40,
                 "augmented_from": None})
    write_manifest(recs, tmp_path / "manifest.jsonl")
    s = _export(tmp_path)
    assert [p["id"] for p in s["problems"]] == ["smartphone/turned"]
    assert s["problems"][0]["reason"].startswith("EXIF-rotated")


ultralytics = pytest.importorskip("ultralytics")


@pytest.fixture
def trained(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    from src.yolo_train import build_parser, train
    _dataset(tmp_path)
    _export(tmp_path)
    args = build_parser().parse_args(["--data", str(tmp_path / "yolo" / "data.yaml"), "--model", "yolo11n.yaml",
                                      "--run_dir", str(tmp_path / "run"), "--epochs", "1", "--imgsz", "64",
                                      "--batch", "4", "--workers", "0", "--device", "cpu", "--no_plots"])
    train(args)
    return tmp_path


def test_train_writes_per_class_test_metrics(trained):
    run = trained / "run"
    assert (run / "weights" / "best.pt").exists() and (run / "classes.json").exists()
    m = json.loads((run / "test_metrics.json").read_text())
    rows = {r["class"]: r for r in m["per_class"]}
    assert set(rows) == {"a_cry", "a_cry_of", "a_cry_of_cs", "na_cs"}
    assert rows["a_cry"]["images"] == 2 and rows["a_cry_of"]["mAP50"] is None


def test_autolabel_writes_labelme_and_never_overwrites(trained):
    from src.autolabel import build_parser, run
    photos = trained / "raw" / "asbestos"
    keep = photos / "test_0.json"
    keep.write_text('{"human": true}')
    args = build_parser().parse_args(["--weights", str(trained / "run" / "weights" / "best.pt"),
                                      "--images", str(photos), "--imgsz", "64", "--conf", "0.0001",
                                      "--device", "cpu", "--class_config", str(MACRO_CONFIG)])
    df = run(args)
    assert json.loads(keep.read_text()) == {"human": True}               # human file untouched
    assert len(df) == len(list(photos.glob("*.jpg"))) - 1
    doc = json.loads((photos / "train_0.json").read_text())
    assert doc["imagePath"] == "train_0.jpg" and (doc["imageWidth"], doc["imageHeight"]) == (64, 64)
    for s in doc["shapes"]:
        assert s["shape_type"] == "rectangle" and s["description"].startswith("auto ")
        assert s["label"] in {"Crysotile", "A-CRY-OF", "A-CRY-OF-CS", "NA-CS"}
    assert (photos / "review.csv").exists()

    # Manifest mode: JSON go to out_dir, image classes are compared with the labels.
    out = trained / "check"
    args = build_parser().parse_args(["--weights", str(trained / "run" / "weights" / "best.pt"),
                                      "--manifest", str(trained / "manifest.jsonl"), "--data_dir", str(trained),
                                      "--split", "test", "--out_dir", str(out), "--imgsz", "64",
                                      "--conf", "0.0001", "--device", "cpu", "--class_config", str(MACRO_CONFIG)])
    df = run(args)
    assert len(df) == 4 and set(df["true_class"]) == {"a_cry", "na_cs"}
    assert (out / "agreement.json").exists() and (out / "image_class_confusion.csv").exists()
