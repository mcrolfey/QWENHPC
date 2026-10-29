"""Manifest I/O. A manifest is a JSON-lines file with one record per image:

    {
      "id": "raw/microscopic/A-CF/S012_03",
      "image": "raw/microscopic/A-CF/S012_03.tif",         # relative to data_dir (or absolute)
      "label": "A-CF",                                     # raw label (mapped by LabelSpace)
      "modality": "microscopic",                           # microscopic | smartphone
      "sample_id": "S012",                                 # physical specimen; used for grouped
                                                           # splits and paired McNemar tests
      "split": "train",                                    # train | val | test
      "width": 4096, "height": 3072,
      "boxes": [{"bbox": [x_min, y_min, x_max, y_max], "label": "A-CF"}],
      "augmented_from": "..."                              # only on augmented copies
    }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def read_manifest(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return records


def write_manifest(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def resolve_image(record: dict, data_dir: str | Path) -> Path:
    p = Path(record["image"])
    return p if p.is_absolute() else Path(data_dir) / p


def summarise(records: list[dict], key_fn=lambda r: r["label"]) -> str:
    table: dict[tuple[str, str], Counter] = {}
    for r in records:
        table.setdefault((r.get("modality", "?"), r.get("split", "?")), Counter())[key_fn(r)] += 1
    lines = []
    for (mod, split), counts in sorted(table.items()):
        total = sum(counts.values())
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"  {mod:<12} {split:<6} n={total:<6} {parts}")
    return "\n".join(lines)
