"""Bounding-box geometry and box-preserving tiling.

Boxes are absolute-pixel Pascal VOC ``[x_min, y_min, x_max, y_max]`` with an
associated label. All functions are pure (no image I/O) so they can be unit
tested and shared between on-the-fly tiling (dataset.py) and offline
augmentation (augment.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def as_list(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_dict(self) -> dict:
        return {"bbox": [round(v, 2) for v in self.as_list()], "label": self.label}

    @classmethod
    def from_dict(cls, d: dict, default_label: str) -> "Box":
        x0, y0, x1, y1 = (float(v) for v in d["bbox"])
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), str(d.get("label", default_label)))


def clip_box(box: Box, width: int, height: int) -> Box | None:
    """Clip a box to image bounds; None if nothing remains."""
    x0, y0 = max(0.0, box.x_min), max(0.0, box.y_min)
    x1, y1 = min(float(width), box.x_max), min(float(height), box.y_max)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1, y1, box.label)


def crop_boxes(
    boxes: list[Box],
    crop: tuple[int, int, int, int],
    min_visibility: float = 0.3,
    min_size: float = 2.0,
) -> list[Box]:
    """Translate boxes into crop coordinates, keeping those sufficiently visible.

    ``crop`` is ``(left, top, right, bottom)``. A box is kept if the fraction
    of its area inside the crop is >= ``min_visibility`` and the clipped box is
    at least ``min_size`` px on both sides.
    """
    left, top, right, bottom = crop
    out = []
    for b in boxes:
        if b.area <= 0:
            continue
        shifted = Box(b.x_min - left, b.y_min - top, b.x_max - left, b.y_max - top, b.label)
        clipped = clip_box(shifted, right - left, bottom - top)
        if clipped is None:
            continue
        if clipped.area / b.area < min_visibility:
            continue
        if (clipped.x_max - clipped.x_min) < min_size or (clipped.y_max - clipped.y_min) < min_size:
            continue
        out.append(clipped)
    return out


def tile_grid(width: int, height: int, tile: int, overlap: float = 0.0) -> list[tuple[int, int, int, int]]:
    """Cover an image with ``tile x tile`` windows (last row/column snapped to edge).

    Images smaller than the tile in a dimension get a single window spanning
    that dimension, so every returned window lies within the image.
    """
    if tile <= 0:
        return [(0, 0, width, height)]
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(round(tile * (1.0 - overlap))))

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        s = list(range(0, size - tile + 1, stride))
        if s[-1] != size - tile:
            s.append(size - tile)
        return s

    windows = []
    for top in starts(height):
        for left in starts(width):
            windows.append((left, top, min(left + tile, width), min(top + tile, height)))
    return windows


def hflip_boxes(boxes: list[Box], width: int) -> list[Box]:
    return [Box(width - b.x_max, b.y_min, width - b.x_min, b.y_max, b.label) for b in boxes]


def vflip_boxes(boxes: list[Box], height: int) -> list[Box]:
    return [Box(b.x_min, height - b.y_max, b.x_max, height - b.y_min, b.label) for b in boxes]


def rot90_boxes(boxes: list[Box], width: int, height: int) -> list[Box]:
    """Boxes after rotating the image 90 degrees counter-clockwise (PIL ROTATE_90).

    A point (x, y) maps to (y, width - x); the new image is ``height x width``.
    """
    return [Box(b.y_min, width - b.x_max, b.y_max, width - b.x_min, b.label) for b in boxes]
