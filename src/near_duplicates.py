"""Near-duplicate check between the training images and the validation/test images.

Splits are made per image, so several photos of the same specimen can end up on both
sides of a split. A model can then score well by recognising the specimen rather than
the material. This script finds such pairs with a perceptual hash (DCT pHash, 64 bits)
and reports how many evaluation images have a close match in the training split.

Each evaluation image is hashed under the 8 flips/90-degree rotations and the smallest
Hamming distance is kept, so a flipped or rotated copy of a training photo is still
found. Augmented records (``augmented_from``) are ignored: they are copies of training
images by design.

Rough reading of the Hamming distance (out of 64): 0-4 the same photo (re-saved,
resized); 5-10 very likely the same scene; above ~14 usually unrelated.

Example::

    python -m src.near_duplicates --data_dir $DATA --modality smartphone \
        --output_dir $RESULTS/near_duplicates_smartphone \
        --predictions $RESULTS/phone_aug_on_smartphone/predictions_reaggregated.csv

With ``--predictions`` the accuracy is also reported separately for test images with
and without a near-duplicate in training: if the score drops on the "clean" images,
the headline result is inflated by near-duplicates.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.manifest import read_manifest, resolve_image

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger("near_duplicates")

HASH_SIZE = 8
IMG_SIZE = 32
TRANSFORMS = ["identity", "rot90", "rot180", "rot270", "flip", "flip_rot90", "flip_rot180", "flip_rot270"]


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


_DCT = _dct_matrix(IMG_SIZE)


def phash_bits(gray: np.ndarray) -> np.ndarray:
    """64-bit pHash of a 32x32 grayscale array, as a bool vector."""
    coeffs = _DCT @ gray @ _DCT.T
    low = coeffs[:HASH_SIZE, :HASH_SIZE].flatten()
    return low > np.median(low[1:])          # the DC term would dominate the median


def dihedral(a: np.ndarray) -> list[np.ndarray]:
    """The 8 flips/rotations of a square array, in the order of TRANSFORMS."""
    out = [np.rot90(a, k) for k in range(4)]
    f = np.fliplr(a)
    return out + [np.rot90(f, k) for k in range(4)]


def hash_image(path: str) -> np.ndarray | None:
    """(8, 64) bool array: the pHash of each flip/rotation, or None if unreadable."""
    try:
        with Image.open(path) as im:
            im.draft("L", (IMG_SIZE * 4, IMG_SIZE * 4))   # fast JPEG decode at reduced size
            small = im.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    except Exception as e:  # noqa: BLE001 - report and skip unreadable files
        log.warning("cannot read %s: %s", path, e)
        return None
    gray = np.asarray(small, dtype=np.float64)
    return np.stack([phash_bits(g) for g in dihedral(gray)])


def hash_all(paths: list[str], workers: int) -> list[np.ndarray | None]:
    if workers <= 1:
        return [hash_image(p) for p in tqdm(paths, desc="hash")]
    with ProcessPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(hash_image, paths, chunksize=32), total=len(paths), desc="hash"))


def nearest(query: np.ndarray, ref: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query image the closest reference image over all 8 transforms.

    query: (n, 8, 64) bool, ref: (m, 64) bool (reference hashed untransformed).
    Returns (distance, reference index, transform index), each of length n.
    """
    r = ref.astype(np.float32)
    r_ones = r.sum(1)
    best_d = np.full(len(query), 65, dtype=np.int64)
    best_j = np.zeros(len(query), dtype=np.int64)
    best_t = np.zeros(len(query), dtype=np.int64)
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk].astype(np.float32)                 # (c, 8, 64)
        # Hamming = |q| + |r| - 2 q.r
        d = q.sum(2)[:, :, None] + r_ones[None, None, :] - 2 * np.einsum("ctk,mk->ctm", q, r)
        d = np.rint(d).astype(np.int64)                            # (c, 8, m)
        flat = d.reshape(len(q), -1).argmin(1)
        t, j = np.divmod(flat, len(ref))
        best_d[s:s + chunk] = d[np.arange(len(q)), t, j]
        best_j[s:s + chunk] = j
        best_t[s:s + chunk] = t
    return best_d, best_j, best_t


def side_by_side(a: Path, b: Path, out: Path, height: int = 384) -> None:
    ims = []
    for p in (a, b):
        with Image.open(p) as im:
            im = im.convert("RGB")
            ims.append(im.resize((max(1, round(im.width * height / im.height)), height)))
    canvas = Image.new("RGB", (ims[0].width + ims[1].width + 8, height), "white")
    canvas.paste(ims[0], (0, 0))
    canvas.paste(ims[1], (ims[0].width + 8, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)


def accuracy_by_match(pred_csv: str, pairs: pd.DataFrame, threshold: int) -> dict:
    """Accuracy on evaluation images with and without a training near-duplicate."""
    pred = pd.read_csv(pred_csv)
    if "task" in pred.columns:
        pred = pred[pred["task"] == "binary"]
    near = set(pairs.loc[pairs["distance"] <= threshold, "record_id"])
    pred = pred[pred["record_id"].isin(set(pairs["record_id"]))]
    out = {"predictions": pred_csv, "threshold": threshold}
    for name, sub in (("with_near_duplicate", pred[pred["record_id"].isin(near)]),
                      ("without_near_duplicate", pred[~pred["record_id"].isin(near)])):
        out[name] = {"n": int(len(sub)),
                     "accuracy": float(sub["correct"].astype(bool).mean()) if len(sub) else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest", default=None, help="Default: <data_dir>/manifest.jsonl")
    ap.add_argument("--modality", choices=["microscopic", "smartphone"], required=True)
    ap.add_argument("--reference_split", default="train")
    ap.add_argument("--query_splits", nargs="+", default=["test", "val"])
    ap.add_argument("--threshold", type=int, default=10,
                    help="Hamming distance (of 64) at or below which a pair counts as a near-duplicate.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--previews", type=int, default=30, help="Side-by-side images of the N closest pairs.")
    ap.add_argument("--predictions", default=None,
                    help="predictions*.csv of a model on the test split: report accuracy with/without a match.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [r for r in read_manifest(args.manifest or data_dir / "manifest.jsonl")
               if r["modality"] == args.modality and "augmented_from" not in r]
    ref = [r for r in records if r.get("split") == args.reference_split]
    query = [r for r in records if r.get("split") in args.query_splits]
    log.info("%s: %d %s images vs %d %s images", args.modality, len(query), "/".join(args.query_splits),
             len(ref), args.reference_split)
    if not ref or not query:
        raise SystemExit("No images to compare - check --modality and the split names.")

    hashes = hash_all([str(resolve_image(r, data_dir)) for r in ref + query], args.workers)
    ok = [h is not None for h in hashes]
    ref_ok = [r for r, k in zip(ref, ok[:len(ref)]) if k]
    query_ok = [r for r, k in zip(query, ok[len(ref):]) if k]
    ref_h = np.stack([h[0] for h in hashes[:len(ref)] if h is not None])
    query_h = np.stack([h for h in hashes[len(ref):] if h is not None])

    dist, j, t = nearest(query_h, ref_h)
    pairs = pd.DataFrame({
        "record_id": [r["id"] for r in query_ok],
        "split": [r["split"] for r in query_ok],
        "label": [r["label"] for r in query_ok],
        "image": [r["image"] for r in query_ok],
        "train_id": [ref_ok[k]["id"] for k in j],
        "train_label": [ref_ok[k]["label"] for k in j],
        "train_image": [ref_ok[k]["image"] for k in j],
        "distance": dist,
        "transform": [TRANSFORMS[k] for k in t],
    }).sort_values("distance")
    pairs["same_label"] = pairs["label"] == pairs["train_label"]
    pairs.to_csv(out_dir / "near_duplicates.csv", index=False)

    summary = {"modality": args.modality, "reference_split": args.reference_split,
               "n_reference": len(ref_ok), "threshold": args.threshold, "splits": {}}
    for split, sub in pairs.groupby("split"):
        near = sub[sub["distance"] <= args.threshold]
        summary["splits"][split] = {
            "n": int(len(sub)),
            "with_near_duplicate": int(len(near)),
            "fraction": round(len(near) / len(sub), 4),
            "same_label_among_near": round(float(near["same_label"].mean()), 4) if len(near) else None,
            "distance_counts": {f"<={d}": int((sub["distance"] <= d).sum()) for d in (0, 2, 4, 6, 8, 10, 12, 14)},
            "median_distance": float(sub["distance"].median()),
        }
    if args.predictions:
        test_pairs = pairs[pairs["split"] == "test"]
        summary["accuracy_by_match"] = accuracy_by_match(args.predictions, test_pairs, args.threshold)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    for i, row in enumerate(pairs.head(args.previews).itertuples()):
        name = f"{i:03d}_d{row.distance:02d}_{row.split}.jpg"
        try:
            side_by_side(resolve_image({"image": row.image}, data_dir),
                         resolve_image({"image": row.train_image}, data_dir), out_dir / "previews" / name)
        except Exception as e:  # noqa: BLE001
            log.warning("preview %s failed: %s", name, e)

    for split, s in summary["splits"].items():
        log.info("%s: %d of %d images (%.1f%%) have a training image within distance %d; "
                 "same label in %s of those; median distance %.0f",
                 split, s["with_near_duplicate"], s["n"], 100 * s["fraction"], args.threshold,
                 s["same_label_among_near"], s["median_distance"])
        log.info("  images with a match at distance <= d: %s", s["distance_counts"])
    if args.predictions:
        a = summary["accuracy_by_match"]
        log.info("Test accuracy with a near-duplicate in training: %s (n=%d); without: %s (n=%d)",
                 a["with_near_duplicate"]["accuracy"], a["with_near_duplicate"]["n"],
                 a["without_near_duplicate"]["accuracy"], a["without_near_duplicate"]["n"])
    log.info("Wrote near_duplicates.csv, summary.json and previews/ to %s - look at the previews "
             "to decide which distances really are the same specimen.", out_dir)


if __name__ == "__main__":
    main()
