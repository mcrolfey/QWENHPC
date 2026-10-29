"""Augmented boxes must still enclose the object they annotated.

The object masks go through the same (geometric) transforms as the image, so
the bounding box of each transformed mask is the ground truth for the
transformed box.
"""

import random

import numpy as np
import pytest

from src.augment import augment_record, build_pipeline, copies_per_record


def _scene(h=480, w=640):
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    img[100:140, 200:420] = 255
    mask[100:140, 200:420] = 1    # elongated "fibre"
    img[300:420, 60:110] = 255
    mask[300:420, 60:110] = 2     # second object
    boxes = [[200, 100, 420, 140], [60, 300, 110, 420]]
    return img, mask, boxes, ["chrysotile", "amosite"]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@pytest.mark.parametrize("modality", ["microscopic", "smartphone"])
@pytest.mark.parametrize("seed", range(25))
def test_boxes_match_transformed_objects(modality, seed):
    img, mask, boxes, labels = _scene()
    pipe = build_pipeline(modality, min_visibility=0.4)
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(pipe, "set_random_seed"):
        pipe.set_random_seed(seed)
    out = pipe(image=img, mask=mask, bboxes=boxes, box_labels=labels)
    assert out["image"].shape[:2] == out["mask"].shape[:2]
    for bb, lab in zip(out["bboxes"], out["box_labels"]):
        obj = out["mask"] == (1 if lab == "chrysotile" else 2)
        ys, xs = np.nonzero(obj)
        truth = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        # 'ellipse' rotation deliberately trims the corners of rotated rectangles,
        # so allow some slack but require a close match.
        assert _iou(bb, truth) > 0.6, (lab, bb, truth)
        x0, y0, x1, y1 = (int(round(v)) for v in bb)
        assert obj[y0:y1, x0:x1].sum() / obj.sum() > 0.8


def test_augment_record_keeps_labels_and_bounds():
    img, _, boxes, labels = _scene()
    rec = {"label": "chrysotile", "boxes": [{"bbox": b, "label": l} for b, l in zip(boxes, labels)]}
    for seed in range(10):
        out, new = augment_record(rec, img, build_pipeline("smartphone", 0.4), seed)
        h, w = out.shape[:2]
        for b in new:
            x0, y0, x1, y1 = b["bbox"]
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
            assert b["label"] in labels
    # same seed -> identical result (reproducible datasets)
    a = augment_record(rec, img, build_pipeline("microscopic", 0.4), 7)
    b = augment_record(rec, img, build_pipeline("microscopic", 0.4), 7)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_balance_gives_minority_more_copies():
    recs = [{"modality": "microscopic"}] * 4
    keys = ["a", "a", "a", "b"]
    n = copies_per_record(recs, keys, copies=1, balance=True)
    assert n[:3] == [1, 1, 1]
    # target = 3 majority images * (1 + 1 copy) = 6 per class -> minority needs 5 copies
    assert n[3] == 5
    assert 1 * (1 + n[3]) == 3 * (1 + 1)
