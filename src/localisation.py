"""Do the microscope model's high-scoring tiles contain the annotated fibres?

Image-level accuracy does not show *where* the model looks: an image can be called
asbestos for the right reason (tiles with fibres score high) or the wrong one. This
script needs no GPU. It reads the tile scores that ``src.evaluate predict`` saved
(``tiles.csv``) and checks them against the bounding boxes in the manifest.

Each tile is labelled from the asbestos boxes it overlaps:

* ``fibre``   - at least ``--min_visibility`` of an asbestos box lies inside the tile
                (the rule used to label training tiles);
* ``partial`` - an asbestos box overlaps the tile, but less than that;
* ``empty``   - no asbestos box overlaps the tile.

Reported (asbestos images only, unless stated):

* **Tile AUC vs negative images**: fibre tiles against every tile of non-asbestos
  images. High = fibre tiles look unlike anything in clean images.
* **Within-image AUC**: fibre vs empty tiles of the same asbestos images (pooled, and
  the mean of per-image AUCs). This is the localisation measure: it ignores anything
  that differs between images (stain, magnification, background).
* **Pointing game**: is the image's highest-scoring tile one that overlaps an asbestos
  box (fibre or partial)? Compared with picking a tile at random; the p-value is from
  10,000 random picks per image. ``top3_precision`` does the same for the 3 highest
  tiles, which the selected rule (mean of the top 3) relies on.

Confidence intervals resample images. Example::

    python -m src.localisation --run_dir $RESULTS/micro_on_microscopic \
        --manifest $DATA/manifest.jsonl --class_config configs/classes_micro.yaml \
        --previews 20

Writes ``localisation.json``, ``tiles_localisation.csv`` (one row per tile) and, with
``--previews``, images with the tile scores as a red overlay, the asbestos boxes in
green and the top tile in yellow (``previews/``, both hits and misses).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.boxes import Box
from src.evaluate import _prob_cols
from src.labels import LabelSpace
from src.manifest import read_manifest, resolve_image

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger("localisation")


def visible_fraction(boxes: list[Box], crop: tuple[int, int, int, int]) -> float:
    """Largest fraction of any box's area that lies inside ``crop`` (0 if none overlaps)."""
    left, top, right, bottom = crop
    best = 0.0
    for b in boxes:
        if b.area <= 0:
            continue
        w = min(b.x_max, right) - max(b.x_min, left)
        h = min(b.y_max, bottom) - max(b.y_min, top)
        if w > 0 and h > 0:
            best = max(best, w * h / b.area)
    return best


def label_tiles(tiles: pd.DataFrame, records: dict[str, dict], space: LabelSpace,
                min_visibility: float) -> pd.DataFrame:
    """Add image_asbestos, visible and tile_class columns to the binary tile rows."""
    rows = []
    for rid, g in tiles.groupby("record_id", sort=False):
        rec = records.get(rid)
        if rec is None:
            continue
        key = space.map(rec["label"])
        if key is None:
            continue
        image_asb = space.is_asbestos(key)
        boxes = []
        for b in rec.get("boxes", []) or []:
            bkey = space.map(b.get("label", rec["label"]))
            if bkey is not None and space.is_asbestos(bkey):
                boxes.append(Box.from_dict(b, bkey))
        for row in g.itertuples(index=False):
            vis = visible_fraction(boxes, tuple(json.loads(row.crop))) if image_asb else 0.0
            cls = "fibre" if vis >= min_visibility and vis > 0 else ("partial" if vis > 0 else "empty")
            rows.append({"record_id": rid, "tile": row.tile, "crop": row.crop, "score": row.score,
                         "image_asbestos": image_asb, "n_asbestos_boxes": len(boxes),
                         "visible": round(vis, 4), "tile_class": cls})
    return pd.DataFrame(rows)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def per_image(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Pointing-game rows for asbestos images with at least one box-overlapping and one empty tile."""
    rows = []
    for rid, g in df[df["image_asbestos"]].groupby("record_id", sort=False):
        hit = (g["tile_class"] != "empty").to_numpy()
        if hit.all() or not hit.any():
            continue
        order = np.argsort(-g["score"].to_numpy(), kind="stable")
        fib, emp = g[g["tile_class"] == "fibre"]["score"], g[g["tile_class"] == "empty"]["score"]
        rows.append({"record_id": rid, "n_tiles": len(g), "n_overlapping": int(hit.sum()),
                     "chance": float(hit.mean()), "top1_hit": bool(hit[order[0]]),
                     f"top{k}_precision": float(hit[order[:k]].mean()),
                     "top_tile": int(g["tile"].iloc[order[0]]),
                     "auc": _auc(fib.to_numpy(), emp.to_numpy())})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, img: pd.DataFrame, n_boot: int, n_sim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pos = df[df["image_asbestos"]]
    fibre, empty = pos[pos["tile_class"] == "fibre"], pos[pos["tile_class"] == "empty"]
    neg = df[~df["image_asbestos"]]

    out = {
        "tiles": {"fibre": int(len(fibre)), "partial": int((pos["tile_class"] == "partial").sum()),
                  "empty_in_asbestos_images": int(len(empty)), "in_non_asbestos_images": int(len(neg))},
        "images": {"asbestos": int(pos["record_id"].nunique()), "non_asbestos": int(neg["record_id"].nunique()),
                   "in_pointing_game": int(len(img))},
        "mean_score": {"fibre": float(fibre["score"].mean()) if len(fibre) else None,
                       "empty_in_asbestos_images": float(empty["score"].mean()) if len(empty) else None,
                       "in_non_asbestos_images": float(neg["score"].mean()) if len(neg) else None},
        "tile_auc_fibre_vs_negative_images": _auc(fibre["score"].to_numpy(), neg["score"].to_numpy()),
        "within_image_auc_pooled": _auc(fibre["score"].to_numpy(), empty["score"].to_numpy()),
    }
    if len(img):
        aucs = img["auc"].dropna()
        hits, chance = img["top1_hit"].to_numpy(float), img["chance"].to_numpy()
        sims = (rng.random((n_sim, len(chance))) < chance).mean(axis=1)
        k = [c for c in img.columns if c.endswith("_precision")][0]
        out["within_image_auc_mean"] = float(aucs.mean()) if len(aucs) else None
        out["within_image_auc_median"] = float(aucs.median()) if len(aucs) else None
        out["pointing_game"] = {
            "top1_hit_rate": float(hits.mean()),
            "chance": float(chance.mean()),
            "p_value": float((1 + (sims >= hits.mean()).sum()) / (1 + n_sim)),
            k: float(img[k].mean()),
        }

    # image-level bootstrap for the headline numbers (tile row indices grouped by image)
    score = df["score"].to_numpy()
    kind = np.where(~df["image_asbestos"].to_numpy(), 0, df["tile_class"].map({"fibre": 1, "empty": 2, "partial": 3}))
    groups = list(df.groupby("record_id", sort=False).indices.values())
    hit_of = dict(zip(img["record_id"], img["top1_hit"])) if len(img) else {}
    hit_by_group = np.array([hit_of.get(rid, np.nan) for rid in df.groupby("record_id", sort=False).indices],
                            dtype=float)
    boots = {"within_image_auc_pooled": [], "tile_auc_fibre_vs_negative_images": [], "top1_hit_rate": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        s, k = score[rows], kind[rows]
        boots["within_image_auc_pooled"].append(_auc(s[k == 1], s[k == 2]))
        boots["tile_auc_fibre_vs_negative_images"].append(_auc(s[k == 1], s[k == 0]))
        h = hit_by_group[pick]
        if (~np.isnan(h)).any():
            boots["top1_hit_rate"].append(float(np.nanmean(h)))
    out["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                   for k, v in ((k, [x for x in v if x is not None]) for k, v in boots.items()) if v}
    return out


def draw_preview(rec: dict, tiles: pd.DataFrame, top_tile: int, data_dir: Path, out: Path,
                 max_side: int = 1024) -> None:
    with Image.open(resolve_image(rec, data_dir)) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_side / max(im.size))
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
    heat = np.zeros((im.height, im.width), np.float32)
    for row in tiles.itertuples(index=False):
        l, t, r, b = (round(v * scale) for v in json.loads(row.crop))
        heat[t:b, l:r] = np.maximum(heat[t:b, l:r], row.score)
    red = Image.new("RGB", im.size, (255, 0, 0))
    im = Image.composite(red, im, Image.fromarray((heat * 140).astype(np.uint8)))
    d = ImageDraw.Draw(im)
    for b in rec.get("boxes", []) or []:
        d.rectangle([v * scale for v in b["bbox"]], outline=(0, 255, 0), width=2)
    top = tiles[tiles["tile"] == top_tile]
    if len(top):
        d.rectangle([v * scale for v in json.loads(top["crop"].iloc[0])], outline=(255, 255, 0), width=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="predict output folder with tiles.csv (a tiled test set).")
    ap.add_argument("--manifest", required=True, help="Manifest the predictions were made from (has the boxes).")
    ap.add_argument("--data_dir", default=None, help="For --previews. Default: the manifest's folder.")
    ap.add_argument("--class_config", default="configs/classes_micro.yaml",
                    help="Decides which image and box labels count as asbestos.")
    ap.add_argument("--min_visibility", type=float, default=0.3,
                    help="Box fraction inside a tile for it to count as a fibre tile (training used 0.3).")
    ap.add_argument("--output_dir", default=None, help="Default: <run_dir>/localisation")
    ap.add_argument("--previews", type=int, default=0, help="Overlay images: N/2 pointing-game hits, N/2 misses.")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir or run_dir / "localisation")
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = pd.read_csv(run_dir / "tiles.csv")
    tiles = tiles[tiles["task"] == "binary"].copy()
    pcols = _prob_cols(tiles)
    if tiles.empty or len(pcols) != 2:
        raise SystemExit(f"{run_dir}/tiles.csv has no binary tile scores")
    if tiles["crop"].isna().all() or (tiles["crop"].astype(str) == "").all():
        raise SystemExit("These predictions are not tiled (whole images): there is nothing to localise.")
    tiles = tiles[tiles["crop"].notna() & (tiles["crop"].astype(str) != "")]
    tiles["score"] = tiles[pcols[1]]  # pcols = [negative, positive]

    records = {r["id"]: r for r in read_manifest(args.manifest)}
    space = LabelSpace.load(args.class_config)
    df = label_tiles(tiles, records, space, args.min_visibility)
    if df.empty:
        raise SystemExit("No tile matched a manifest record - is --manifest the one used for prediction?")
    df.to_csv(out_dir / "tiles_localisation.csv", index=False)

    img = per_image(df)
    summary = {"run_dir": str(run_dir), "min_visibility": args.min_visibility,
               **summarise(df, img, args.bootstrap, 10_000, args.seed)}
    (out_dir / "localisation.json").write_text(json.dumps(summary, indent=2))
    if len(img):
        img.to_csv(out_dir / "pointing_game.csv", index=False)

    if args.previews and len(img):
        data_dir = Path(args.data_dir or Path(args.manifest).parent)
        n = args.previews // 2
        hits = img[img["top1_hit"]].sort_values("auc", ascending=False).head(n)
        misses = img[~img["top1_hit"]].sort_values("auc").head(args.previews - len(hits))
        by_id = {rid: g for rid, g in df.groupby("record_id", sort=False)}
        for kind, sel in (("hit", hits), ("miss", misses)):
            for i, row in enumerate(sel.itertuples(index=False)):
                auc = float("nan") if row.auc is None else row.auc
                name = f"{kind}_{i:02d}_auc{auc:.2f}.jpg"
                try:
                    draw_preview(records[row.record_id], by_id[row.record_id], row.top_tile, data_dir,
                                 out_dir / "previews" / name)
                except Exception as e:  # noqa: BLE001
                    log.warning("preview %s failed: %s", name, e)

    t, s = summary["tiles"], summary
    log.info("Tiles: %d fibre, %d partial, %d empty in asbestos images, %d in non-asbestos images",
             t["fibre"], t["partial"], t["empty_in_asbestos_images"], t["in_non_asbestos_images"])
    log.info("Mean P(asbestos): fibre %s | empty, asbestos images %s | non-asbestos images %s",
             *(f"{v:.3f}" if v is not None else "-" for v in s["mean_score"].values()))
    log.info("Tile AUC, fibre tiles vs non-asbestos images: %s %s", _fmt(s["tile_auc_fibre_vs_negative_images"]),
             s["ci95"].get("tile_auc_fibre_vs_negative_images", ""))
    log.info("Within-image AUC, fibre vs empty tiles: pooled %s %s, mean per image %s",
             _fmt(s["within_image_auc_pooled"]), s["ci95"].get("within_image_auc_pooled", ""),
             _fmt(s.get("within_image_auc_mean")))
    if "pointing_game" in s:
        pg = s["pointing_game"]
        log.info("Pointing game (%d images): top tile overlaps a box in %.3f %s vs %.3f by chance (p = %.4g); "
                 "top-3 precision %.3f", s["images"]["in_pointing_game"], pg["top1_hit_rate"],
                 s["ci95"].get("top1_hit_rate", ""), pg["chance"], pg["p_value"], pg["top3_precision"])
    log.info("Wrote localisation.json, tiles_localisation.csv and pointing_game.csv to %s", out_dir)


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
