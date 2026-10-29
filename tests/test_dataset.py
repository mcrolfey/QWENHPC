import numpy as np
import torch
from PIL import Image

from conftest import MINERAL_CONFIG as DEFAULT_CONFIG
from src.dataset import AsbestosVLMDataset, DataConfig, VLMCollator
from src.labels import LabelSpace


def _make(tmp_path):
    Image.fromarray(np.zeros((1024, 1024, 3), np.uint8)).save(tmp_path / "m.png")
    Image.fromarray(np.zeros((300, 400, 3), np.uint8)).save(tmp_path / "p.jpg")
    return [
        {"id": "m", "image": "m.png", "label": "chrysotile", "modality": "microscopic", "sample_id": "S1",
         "split": "train", "width": 1024, "height": 1024,
         "boxes": [{"bbox": [10, 10, 100, 30], "label": "white asbestos"}]},
        {"id": "p", "image": "p.jpg", "label": "NAD", "modality": "smartphone", "sample_id": "S2",
         "split": "train", "width": 400, "height": 300, "boxes": []},
        {"id": "p_aug", "image": "p.jpg", "label": "NAD", "modality": "smartphone", "sample_id": "S2",
         "split": "test", "width": 400, "height": 300, "boxes": [], "augmented_from": "p"},
    ]


def test_tiling_with_box_labels(tmp_path):
    space = LabelSpace.load(DEFAULT_CONFIG)
    cfg = DataConfig(modality="unified", tasks=("binary", "type"), tile_size=512,
                     tile_label_source="boxes", empty_tile_policy="drop")
    ds = AsbestosVLMDataset(_make(tmp_path), tmp_path, space, cfg, split="train", train=True)
    micro = [it for it in ds.items if ds.records[it.rec_idx]["modality"] == "microscopic"]
    # 4 tiles; only the top-left tile has the fibre, the other 3 are ambiguous -> dropped
    assert {it.tile_idx for it in micro} == {0}
    assert all(it.key == "chrysotile" for it in micro)
    phone = [it for it in ds.items if ds.records[it.rec_idx]["modality"] == "smartphone"]
    assert len(phone) == 2 and all(it.crop is None for it in phone)
    ex = ds[0]
    assert ex["image"].size == (512, 512)
    assert ex["answer"] in ("Yes", "Chrysotile")
    assert "microscope" in ex["prompt"]


def test_negative_policy_and_eval_excludes_augmented(tmp_path):
    space = LabelSpace.load(DEFAULT_CONFIG)
    cfg = DataConfig(modality="microscopic", tasks=("type",), tile_size=512,
                     tile_label_source="boxes", empty_tile_policy="negative")
    ds = AsbestosVLMDataset(_make(tmp_path), tmp_path, space, cfg, split="train", train=True)
    assert sorted(it.key for it in ds.items) == ["chrysotile"] + ["non_asbestos"] * 3
    test = AsbestosVLMDataset(_make(tmp_path), tmp_path, space, DataConfig(modality="smartphone"),
                              split="test", train=False)
    assert len(test) == 0


class _Tok:
    pad_token_id = 0
    eos_token_id = 0
    def encode(self, s, add_special_tokens=False):
        return [7, 8, 9]  # stands in for "<|im_start|>assistant\n"


class _Proc:
    tokenizer = _Tok()


def test_label_masking_only_keeps_answer():
    col = VLMCollator(_Proc(), "train")
    ids = torch.tensor([[1, 7, 8, 9, 5, 7, 8, 9, 42, 43, 2, 0, 0],
                        [7, 8, 9, 44, 2, 0, 0, 0, 0, 0, 0, 0, 0]])
    mask = (ids != 0).long()
    mask[0, :11] = 1
    labels = col.make_labels(ids, mask)
    assert labels[0].tolist() == [-100] * 8 + [42, 43, 2, -100, -100]
    assert labels[1].tolist() == [-100] * 3 + [44, 2] + [-100] * 8
