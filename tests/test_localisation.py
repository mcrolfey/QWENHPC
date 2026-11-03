import json
import sys

import pandas as pd
from PIL import Image

from src.boxes import Box
from src.localisation import visible_fraction
from src.manifest import write_manifest

WINDOWS = [(0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024)]


def test_visible_fraction():
    b = [Box(400, 100, 600, 200, "A-CF")]                    # straddles tiles 0 and 1
    assert visible_fraction(b, WINDOWS[0]) == 0.56            # 112 of 200 px wide
    assert visible_fraction(b, WINDOWS[2]) == 0.0


def _write_run(tmp_path, scores: dict[str, list[float]]):
    rows = []
    for rid, s in scores.items():
        for ti, (win, p) in enumerate(zip(WINDOWS, s)):
            rows.append({"record_id": rid, "task": "binary", "tile": ti, "crop": json.dumps(win), "pred": int(p > .5),
                         "raw_output": "", "p::No": 1 - p, "p::Yes": p})
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(rows).to_csv(run / "tiles.csv", index=False)
    return run


def test_cli_scores_fibre_tiles(tmp_path, monkeypatch):
    from src import localisation
    recs, scores = [], {}
    for i in range(6):                                        # asbestos: fibre in tile i % 4, which scores highest
        t = i % 4
        x, y = WINDOWS[t][:2]
        recs.append({"id": f"a{i}", "image": f"a{i}.png", "label": "A-CF", "modality": "microscopic",
                     "split": "test", "width": 1024, "height": 1024,
                     "boxes": [{"bbox": [x + 100, y + 100, x + 200, y + 200], "label": "A-CF"}]})
        scores[f"a{i}"] = [0.9 if k == t else 0.2 for k in range(4)]
        Image.new("RGB", (1024, 1024), (200, 200, 200)).save(tmp_path / f"a{i}.png")
    for i in range(3):                                        # non-asbestos images: all tiles low
        recs.append({"id": f"n{i}", "image": f"n{i}.png", "label": "NA-OF", "modality": "microscopic",
                     "split": "test", "width": 1024, "height": 1024,
                     "boxes": [{"bbox": [10, 10, 50, 50], "label": "NA-OF"}]})
        scores[f"n{i}"] = [0.1, 0.15, 0.05, 0.1]
    scores["a5"] = [0.95, 0.1, 0.2, 0.2]                      # a5: fibre in tile 1, but tile 0 scores highest
    write_manifest(recs, tmp_path / "manifest.jsonl")
    run = _write_run(tmp_path, scores)
    monkeypatch.setattr(sys, "argv", ["x", "--run_dir", str(run), "--manifest", str(tmp_path / "manifest.jsonl"),
                                      "--bootstrap", "50", "--previews", "2"])
    localisation.main()
    s = json.loads((run / "localisation" / "localisation.json").read_text())
    assert s["tiles"] == {"fibre": 6, "partial": 0, "empty_in_asbestos_images": 18, "in_non_asbestos_images": 12}
    assert s["images"] == {"asbestos": 6, "non_asbestos": 3, "in_pointing_game": 6}
    pg = s["pointing_game"]
    assert pg["chance"] == 0.25 and abs(pg["top1_hit_rate"] - 5 / 6) < 1e-9   # a5's top tile is empty
    assert abs(s["tile_auc_fibre_vs_negative_images"] - 66 / 72) < 1e-9   # a5's fibre tile (0.1) beats 3, ties 6
    assert 0.5 < s["within_image_auc_pooled"] < 1.0
    assert set(s["ci95"]) == {"within_image_auc_pooled", "tile_auc_fibre_vs_negative_images", "top1_hit_rate"}
    tiles = pd.read_csv(run / "localisation" / "tiles_localisation.csv")
    assert (tiles[~tiles["image_asbestos"]]["tile_class"] == "empty").all()   # NA boxes are not fibres
    names = sorted(p.name for p in (run / "localisation" / "previews").glob("*.jpg"))
    assert len(names) == 2 and names[0].startswith("hit_") and names[1].startswith("miss_")


def test_untiled_predictions_are_refused(tmp_path, monkeypatch):
    from src import localisation
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame([{"record_id": "a", "task": "binary", "tile": -1, "crop": "", "pred": 1, "raw_output": "",
                   "p::No": .1, "p::Yes": .9}]).to_csv(run / "tiles.csv", index=False)
    write_manifest([{"id": "a", "image": "a.png", "label": "A-CF", "modality": "smartphone", "split": "test"}],
                   tmp_path / "m.jsonl")
    monkeypatch.setattr(sys, "argv", ["x", "--run_dir", str(run), "--manifest", str(tmp_path / "m.jsonl")])
    try:
        localisation.main()
    except SystemExit as e:
        assert "not tiled" in str(e)
    else:
        raise AssertionError("expected SystemExit")
