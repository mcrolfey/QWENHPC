"""PyTorch dataset of image-prompt-answer triples for VLM fine-tuning.

* ``modality``: ``microscopic``, ``smartphone`` or ``unified`` (both).
* ``tasks``: ``binary`` (asbestos present?) and/or ``type`` (which mineral).
  With both tasks every image/tile yields one example per task.
* Patch tiling: high-resolution microscope images are cut into
  ``tile_size x tile_size`` windows (optionally overlapping) so the model sees
  fibres at native resolution instead of a heavily down-scaled image.
  Bounding boxes are cropped together with the tiles, so tiles can be
  labelled from the boxes they contain (``tile_label_source="boxes"``) rather
  than inheriting a possibly wrong image-level label.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.boxes import Box, crop_boxes, tile_grid
from src.labels import LabelSpace
from src.manifest import read_manifest, resolve_image

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert asbestos analyst. You examine images of building materials "
    "and mineral fibres and classify them precisely."
)
MODALITY_DESCRIPTIONS = {
    "microscopic": "microscope image of a material sample",
    "smartphone": "smartphone photograph of a building material",
}
TILE_PREFIX = "region cropped from a "


@dataclass
class DataConfig:
    modality: str = "unified"                      # microscopic | smartphone | unified
    tasks: tuple[str, ...] = ("binary", "type")
    tile_size: int = 0                             # 0 disables tiling
    tile_overlap: float = 0.0
    tile_modalities: tuple[str, ...] = ("microscopic",)
    max_tiles_per_image: int = 0                   # 0 = unlimited
    tile_label_source: str = "image"               # image | boxes
    empty_tile_policy: str = "drop"                # drop | negative | inherit (positive image, no box in tile)
    min_box_visibility: float = 0.3
    modality_hint: bool = True                     # mention image type in the prompt
    online_aug: bool = False                       # random flips / 90-degree rotations (train only)
    shuffle_options: bool = False                  # shuffle option order in type prompts (train only)
    exclude_augmented: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tasks"] = list(self.tasks)
        d["tile_modalities"] = list(self.tile_modalities)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("tasks", "tile_modalities"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)


def build_prompt(task: str, modality: str, space: LabelSpace, tiled: bool, hint: bool,
                 option_order: list[str] | None = None) -> str:
    desc = MODALITY_DESCRIPTIONS[modality] if hint else "image"
    if tiled and hint:
        desc = TILE_PREFIX + desc
    if task == "binary":
        return (f"Does this {desc} show asbestos? "
                f"Answer with exactly one word: '{space.binary_positive}' or '{space.binary_negative}'.")
    if task == "type":
        options = option_order or space.names
        return (f"Which category best describes the material in this {desc}? "
                f"Options: {', '.join(options)}. Answer with exactly one option, nothing else.")
    raise ValueError(task)


def build_messages(prompt: str, answer: str | None) -> list[dict]:
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


@dataclass
class Item:
    rec_idx: int
    task: str
    key: str                                  # canonical class key of the target
    crop: tuple[int, int, int, int] | None    # (left, top, right, bottom) or None for full image
    tile_idx: int = -1
    n_boxes: int = 0
    meta: dict = field(default_factory=dict)


class AsbestosVLMDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | list[dict],
        data_dir: str | Path,
        space: LabelSpace,
        cfg: DataConfig,
        split: str | None,
        train: bool = False,
        seed: int = 42,
        image_cache: int = 2,
        truth_space: LabelSpace | None = None,
    ) -> None:
        """``space``: classes the model answers with (prompts, answers).
        ``truth_space``: maps manifest labels to ground-truth classes; defaults to
        ``space``. They differ in cross-domain evaluation, where the test images
        come from a dataset with other type classes."""
        self.data_dir = Path(data_dir)
        self.space = space
        self.truth = truth_space or space
        self.cfg = cfg
        self.train = train
        self.seed = seed
        records = read_manifest(manifest) if not isinstance(manifest, list) else manifest
        self.records = [r for r in records if self._keep(r, split)]
        self.items: list[Item] = []
        self._cache: OrderedDict[int, Image.Image] = OrderedDict()
        self._cache_size = image_cache
        self._build_items()

    # --------------------------------------------------------------- indexing
    def _keep(self, r: dict, split: str | None) -> bool:
        if split is not None and r.get("split") != split:
            return False
        if self.cfg.modality != "unified" and r["modality"] != self.cfg.modality:
            return False
        if self.cfg.exclude_augmented and "augmented_from" in r:
            return False
        if not self.train and "augmented_from" in r:
            return False  # never evaluate on synthetic copies
        return self.truth.map(r["label"]) is not None

    def _size(self, r: dict) -> tuple[int, int]:
        if "width" in r and "height" in r:
            return int(r["width"]), int(r["height"])
        with Image.open(resolve_image(r, self.data_dir)) as im:
            return im.size

    def _negative_key(self) -> str | None:
        return next((c.key for c in self.truth.classes if not c.asbestos), None)

    def _tile_key(self, image_key: str, boxes: list[Box]) -> str | None:
        """Target class of a tile given the (mapped) boxes visible in it."""
        if self.cfg.tile_label_source == "image":
            return image_key
        if boxes:
            area: Counter = Counter()
            for b in boxes:
                area[b.label] += b.area
            asb = {k: a for k, a in area.items() if self.truth.is_asbestos(k)}
            pool = asb or area
            return max(pool, key=pool.get)
        if not self.truth.is_asbestos(image_key):
            return image_key
        policy = self.cfg.empty_tile_policy
        if policy == "inherit":
            return image_key
        if policy == "negative":
            return self._negative_key()
        return None  # drop: positive image, no fibre in this tile -> ambiguous

    def _build_items(self) -> None:
        rng = random.Random(self.seed)
        dropped_tiles = 0
        for ri, r in enumerate(self.records):
            image_key = self.truth.map(r["label"])
            width, height = self._size(r)
            boxes = []
            for b in r.get("boxes", []) or []:
                bkey = self.truth.map(b.get("label", r["label"]))  # merges/renames apply to boxes too
                if bkey is not None:
                    boxes.append(Box.from_dict({**b, "label": bkey}, bkey))
            annotated = bool(r.get("boxes"))
            tiled = self.cfg.tile_size > 0 and r["modality"] in self.cfg.tile_modalities
            if not tiled:
                for task in self.cfg.tasks:
                    self.items.append(Item(ri, task, image_key, None, -1, len(boxes)))
                continue

            windows = tile_grid(width, height, self.cfg.tile_size, self.cfg.tile_overlap)
            tiles = []
            for ti, win in enumerate(windows):
                tboxes = crop_boxes(boxes, win, self.cfg.min_box_visibility)
                key = self._tile_key(image_key, tboxes) if annotated else image_key
                if key is None:
                    dropped_tiles += 1
                    continue
                tiles.append((ti, win, key, len(tboxes)))
            cap = self.cfg.max_tiles_per_image
            if self.train and cap and len(tiles) > cap:
                with_boxes = [t for t in tiles if t[3] > 0]
                without = [t for t in tiles if t[3] == 0]
                rng.shuffle(with_boxes)
                rng.shuffle(without)
                tiles = sorted((with_boxes + without)[:cap])
            for ti, win, key, nb in tiles:
                for task in self.cfg.tasks:
                    self.items.append(Item(ri, task, key, win, ti, nb))
        if dropped_tiles:
            log.info("Dropped %d ambiguous tiles (positive image, no annotated fibre in tile)", dropped_tiles)

    # ---------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.items)

    def _load(self, ri: int) -> Image.Image:
        if ri in self._cache:
            self._cache.move_to_end(ri)
            return self._cache[ri]
        with Image.open(resolve_image(self.records[ri], self.data_dir)) as im:
            img = im.convert("RGB")
        self._cache[ri] = img
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        rec = self.records[it.rec_idx]
        img = self._load(it.rec_idx)
        if it.crop is not None:
            img = img.crop(it.crop)
        if self.train and self.cfg.online_aug:
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if rec["modality"] == "microscopic" and random.random() < 0.5:
                img = img.transpose(Image.Transpose.ROTATE_90)
        options = None
        if it.task == "type" and self.train and self.cfg.shuffle_options:
            options = list(self.space.names)
            random.shuffle(options)
        prompt = build_prompt(it.task, rec["modality"], self.space, it.crop is not None,
                              self.cfg.modality_hint, options)
        answer, target = self.target(it)
        return {
            "image": img,
            "prompt": prompt,
            "answer": answer,
            "task": it.task,
            "target": target,
            "index": idx,
        }

    def target(self, it: Item) -> tuple[str | None, int]:
        """Answer text and index in the model's classes; (None, -1) if the true
        class is not one the model knows (cross-domain type task)."""
        if it.task == "binary":
            idx = int(self.truth.is_asbestos(it.key))
        elif it.key in self.space.keys:
            idx = self.space.keys.index(it.key)
        else:
            return None, -1
        return self.space.answers(it.task)[idx], idx

    def sample_weights(self) -> torch.Tensor:
        """Inverse-frequency weights per item (per task and target) for balanced sampling."""
        keys = [(it.task, self.target(it)[1]) for it in self.items]
        counts = Counter(keys)
        return torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)

    def summary(self) -> str:
        c = Counter((self.records[it.rec_idx]["modality"], it.task,
                     self.target(it)[0] or self.truth.by_key(it.key).name) for it in self.items)
        lines = [f"{len(self.records)} images -> {len(self.items)} examples"]
        lines += [f"  {m:<12} {t:<7} {a:<20} {n}" for (m, t, a), n in sorted(c.items())]
        return "\n".join(lines)


class VLMCollator:
    """Turns dataset items into Qwen2.5-VL model inputs.

    ``mode="train"``/``"score"``: prompt + answer, with ``labels`` masking every
    token before the assistant answer (and padding) with -100.
    ``mode="generate"``: prompt only with the generation prompt, left padded.
    """

    def __init__(self, processor, mode: str = "train") -> None:
        if mode not in ("train", "score", "generate"):
            raise ValueError(mode)
        self.processor = processor
        self.mode = mode
        tok = processor.tokenizer
        self.marker = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, items: list[dict]) -> dict:
        texts, images = [], []
        gen = self.mode == "generate"
        for it in items:
            msgs = build_messages(it["prompt"], None if gen else it["answer"])
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen))
            images.append(it["image"])
        self.processor.tokenizer.padding_side = "left" if gen else "right"
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        if not gen:
            batch["labels"] = self.make_labels(batch["input_ids"], batch["attention_mask"])
        return batch

    def make_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = torch.full_like(input_ids, -100)
        m = len(self.marker)
        marker = torch.tensor(self.marker, dtype=input_ids.dtype)
        for row in range(input_ids.size(0)):
            ids = input_ids[row]
            windows = ids.unfold(0, m, 1)
            hits = (windows == marker).all(dim=1).nonzero(as_tuple=True)[0]
            if len(hits) == 0:
                raise RuntimeError("Assistant marker not found; check the chat template.")
            start = int(hits[-1]) + m
            keep = attention_mask[row].bool()
            keep[:start] = False
            labels[row][keep] = ids[keep]
        return labels
