import numpy as np
from PIL import Image

from src.boxes import Box, crop_boxes, hflip_boxes, rot90_boxes, tile_grid, vflip_boxes


def test_tile_grid_covers_image_and_stays_inside():
    w, h, t = 1300, 700, 512
    wins = tile_grid(w, h, t, overlap=0.1)
    covered = np.zeros((h, w), bool)
    for l, tp, r, b in wins:
        assert 0 <= l < r <= w and 0 <= tp < b <= h
        assert r - l == t and b - tp == t
        covered[tp:b, l:r] = True
    assert covered.all()


def test_tile_grid_small_image():
    assert tile_grid(300, 200, 512) == [(0, 0, 300, 200)]


def test_crop_boxes_visibility_and_shift():
    boxes = [Box(100, 100, 200, 120, "chrysotile"), Box(500, 500, 600, 600, "amosite")]
    out = crop_boxes(boxes, (150, 90, 662, 602), min_visibility=0.3)
    # first box: 50% visible -> kept and shifted; second: fully inside
    assert out[0] == Box(0, 10, 50, 30, "chrysotile")
    assert out[1] == Box(350, 410, 450, 510, "amosite")
    assert crop_boxes(boxes, (180, 0, 692, 512), min_visibility=0.3) == []  # only 20% of box 1


def _mask(w, h, box):
    m = np.zeros((h, w), np.uint8)
    m[int(box.y_min):int(box.y_max), int(box.x_min):int(box.x_max)] = 255
    return Image.fromarray(m)


def _bbox_of(img):
    ys, xs = np.nonzero(np.asarray(img))
    return Box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, "x")


def test_flip_and_rotate_match_pil():
    w, h = 64, 40
    b = Box(5, 7, 20, 30, "x")
    img = _mask(w, h, b)
    assert hflip_boxes([b], w)[0] == _bbox_of(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    assert vflip_boxes([b], h)[0] == _bbox_of(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    assert rot90_boxes([b], w, h)[0] == _bbox_of(img.transpose(Image.Transpose.ROTATE_90))
