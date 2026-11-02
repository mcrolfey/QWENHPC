import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

from src.manifest import write_manifest
from src.near_duplicates import dihedral, hash_image, nearest


def _scene(seed, size=(320, 240)):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)            # coarse random blocks
    return Image.fromarray(a).resize(size, Image.BILINEAR)


def test_hash_finds_flipped_rotated_and_resized_copy(tmp_path):
    _scene(0).save(tmp_path / "a.png")
    _scene(0).transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True).resize((200, 260)).save(tmp_path / "a2.jpg", quality=80)
    _scene(1).save(tmp_path / "b.png")
    ref = np.stack([hash_image(str(tmp_path / "a.png"))[0], hash_image(str(tmp_path / "b.png"))[0]])
    query = np.stack([hash_image(str(tmp_path / "a2.jpg")), hash_image(str(tmp_path / "b.png"))])
    d, j, _ = nearest(query, ref, chunk=1)
    assert j.tolist() == [0, 1] and d[1] == 0 and d[0] <= 6
    assert nearest(query[:1], ref[1:])[0][0] > 14                     # unrelated scene is far away


def test_dihedral_has_eight_distinct_views():
    a = np.arange(16).reshape(4, 4)
    assert len({v.tobytes() for v in dihedral(a)}) == 8


def test_cli_reports_matches_and_accuracy(tmp_path, monkeypatch):
    from src import near_duplicates
    recs = []
    for i in range(4):
        _scene(10 + i).save(tmp_path / f"tr{i}.png")
        recs.append({"id": f"tr{i}", "image": f"tr{i}.png", "split": "train", "modality": "smartphone", "label": "A-CRY"})
    _scene(10).resize((300, 225)).save(tmp_path / "te0.jpg")            # copy of tr0
    _scene(99).save(tmp_path / "te1.png")                               # new scene
    recs += [{"id": "te0", "image": "te0.jpg", "split": "test", "modality": "smartphone", "label": "A-CRY"},
             {"id": "te1", "image": "te1.png", "split": "test", "modality": "smartphone", "label": "NA-CS"},
             {"id": "tr0__aug00", "image": "te1.png", "split": "train", "modality": "smartphone",
              "label": "A-CRY", "augmented_from": "tr0"}]                 # augmented: must be ignored
    write_manifest(recs, tmp_path / "manifest.jsonl")
    pd.DataFrame({"record_id": ["te0", "te1"], "task": "binary", "correct": [True, False]}).to_csv(tmp_path / "p.csv", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--data_dir", str(tmp_path), "--modality", "smartphone",
                                      "--output_dir", str(out), "--predictions", str(tmp_path / "p.csv"),
                                      "--workers", "1", "--previews", "2"])
    near_duplicates.main()
    s = json.loads((out / "summary.json").read_text())
    assert s["splits"]["test"]["with_near_duplicate"] == 1 and s["splits"]["test"]["same_label_among_near"] == 1.0
    acc = s["accuracy_by_match"]
    assert acc["with_near_duplicate"] == {"n": 1, "accuracy": 1.0}
    assert acc["without_near_duplicate"] == {"n": 1, "accuracy": 0.0}
    pairs = pd.read_csv(out / "near_duplicates.csv")
    assert pairs.iloc[0]["record_id"] == "te0" and pairs.iloc[0]["train_id"] == "tr0"
    assert len(list((out / "previews").glob("*.jpg"))) == 2


def test_regroup_keeps_near_duplicates_in_one_split(tmp_path, monkeypatch):
    from src import near_duplicates
    from src.manifest import read_manifest
    recs = []
    for i in range(20):                                   # 10 scenes, each photographed twice
        scene, label = i // 2, ("A-CRY" if i < 10 else "NA-CS")
        img = _scene(200 + scene)
        if i % 2:
            img = img.transpose(Image.FLIP_TOP_BOTTOM).resize((300, 230))
        img.save(tmp_path / f"m{i}.png")
        recs.append({"id": f"m{i}", "image": f"m{i}.png", "modality": "smartphone", "label": label,
                     "sample_id": f"smartphone/m{i}", "split": ["train", "test"][i % 2]})   # every pair split apart
    recs.append({"id": "u0", "image": "m0.png", "modality": "microscopic", "label": "A-CF",
                 "sample_id": "microscopic/u0", "split": "val"})
    recs.append({"id": "m0__aug00", "image": "m0.png", "modality": "smartphone", "label": "A-CRY",
                 "sample_id": "smartphone/m0", "split": "train", "augmented_from": "m0"})
    write_manifest(recs, tmp_path / "manifest.jsonl")
    monkeypatch.setattr(sys, "argv", ["x", "--data_dir", str(tmp_path), "--modality", "smartphone",
                                      "--output_dir", str(tmp_path / "out"), "--threshold", "4",
                                      "--workers", "1", "--regroup", "manifest_grouped.jsonl",
                                      "--val_frac", "0.2", "--test_frac", "0.2"])
    near_duplicates.main()
    out = {r["id"]: r for r in read_manifest(tmp_path / "manifest_grouped.jsonl")}
    assert "m0__aug00" not in out and out["u0"]["split"] == "val"          # other modality untouched
    for k in range(0, 20, 2):
        a, b = out[f"m{k}"], out[f"m{k + 1}"]
        assert a["split"] == b["split"] and a["sample_id"] == b["sample_id"]
    assert {r["split"] for r in out.values() if r["modality"] == "smartphone"} == {"train", "val", "test"}
    s = json.loads((tmp_path / "out" / "regroup.json").read_text())
    assert s["groups"] == 10 and s["crossing_pairs"] == 0
