# Data

Images are **not** stored in git. Keep them on the cluster's high-speed project
storage (`$DATA_DIR`, default `/scratch/$USER/asbestos-vlm/data`; on
OzSTAR / Ngarrgu Tindebeek typically `/fred/<project>/$USER/asbestos-vlm/data`).

## 1. Raw layout

```
$DATA_DIR/
└── raw/
    ├── microscopic/
    │   ├── chrysotile/      S012_m01.tif  S012_m01.txt  ...
    │   ├── amosite/
    │   ├── crocidolite/
    │   └── non_asbestos/
    └── smartphone/
        ├── chrysotile/      S012_p01.jpg  ...
        ├── amosite/
        ├── crocidolite/
        └── non_asbestos/
```

* Folder names are **raw labels**. They are mapped to classes through the
  class config (`configs/classes_default.yaml`), so `white_asbestos/`,
  `Chrysotile/` and `chrysotile/` all work, and several folders can be merged
  into one class.
* **Specimen ids matter.** Put the physical sample id in the file name
  (e.g. `S012_…`) and pass `--sample_id_regex '^(S\d+)'`. All images of a
  specimen (microscope *and* phone) then go into the same split, and the
  microscopic and smartphone predictions can be paired for McNemar's test.
* **Bounding boxes (optional)** – YOLO `.txt` next to the image
  (`class cx cy w h`, normalised), with class ids resolved via
  `--yolo_names names.txt` (one raw label per line) or a YOLO `data.yaml`.
  Boxes are used to label microscope tiles and are kept correct by the
  augmentation.

Alternatively provide a CSV with columns `image,label,modality[,sample_id,split]`
(`--csv`), with `image` relative to `$DATA_DIR`.

## 2. Build the manifest

```bash
python -m src.prepare_data --data_dir $DATA_DIR \
    --sample_id_regex '^(S\d+)' --yolo_names $DATA_DIR/names.txt \
    --val_frac 0.15 --test_frac 0.15
```

Writes `$DATA_DIR/manifest.jsonl` (format documented in `src/manifest.py`).

## 3. Augment (training split only)

```bash
python -m src.augment --data_dir $DATA_DIR --copies 3 --balance --preview 20
```

Writes augmented images to `$DATA_DIR/augmented/` and
`$DATA_DIR/manifest_aug.jsonl` (originals + copies). Inspect
`$DATA_DIR/augmented/_preview/` to confirm boxes follow the fibres.
