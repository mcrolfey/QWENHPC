# Data

Images are **not** stored in git. Keep them on the cluster's high-speed project
storage (`$DATA_DIR`, default `/scratch/$USER/asbestos-vlm/data`; on
OzSTAR / Ngarrgu Tindebeek typically `/fred/<project>/$USER/asbestos-vlm/data`).

## 1. Raw layout

```
$DATA_DIR/
└── raw/
    ├── microscopic/
    │   ├── A-AM/      S012_m01.tif  S012_m01.txt  ...
    │   ├── A-CF/
    │   ├── A-COF/
    │   ├── A-CP/
    │   ├── A-CRO/
    │   ├── NA-CS/
    │   └── NA-OF/
    └── smartphone/
        ├── A-CRY/        S012_p01.jpg  ...       (macro dataset)
        ├── A-CRY-OF/
        ├── A-CRY-OF-CS/
        └── NA-CS/
```

* Folder names are **raw labels**. They are mapped to classes through the
  class configs (`configs/classes_micro.yaml` for microscopic,
  `configs/classes_macro.yaml` for the macro/smartphone images). Matching ignores case and
  treats `-`, `_` and spaces alike, so `A-COF/`, `a_cof/` and `a-cof/` all
  work. Other spellings can be added under `aliases`, which is also how
  several folders are merged into one class.
* YOLO class names in `names.txt` use the same codes (`A-AM`, ...).
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

### Importing existing folders in place

Datasets that already sit elsewhere are read where they are, with nothing
copied, and the manifest stores absolute paths:

* **Flat YOLO** (microscopic): `--micro_images <dir> --micro_labels <dir>`.
  Class names are read from `classes.txt`, `obj.names`, `data.yaml` or
  `notes.json` next to the labels (or pass `--micro_names`).
* **JSON folders** (macro): `--macro_dir <dir>` containing `asbestos/` and
  `non-asbestos/`, each with images and JSON boxes. LabelMe (one JSON per
  image), COCO and Label Studio exports are detected automatically.

The image class comes from its boxes: the asbestos class with the largest box
area, or else the largest non-asbestos class. Skipped images (no annotation,
or boxes that contradict the asbestos/non-asbestos folder) are listed in
`$DATA_DIR/prepare_report.csv`. Run with `--dry_run` first. It lists every
label it found and what it maps to, without writing a manifest.

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
