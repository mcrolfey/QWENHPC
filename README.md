# asbestos-vlm-hpc

Fine-tuning and evaluating a vision language model (Qwen2.5-VL-7B-Instruct,
QLoRA) for **asbestos detection** and **asbestos type identification** on
Swinburne's OzSTAR / Ngarrgu Tindebeek Slurm cluster.

> **Research question:** How does the classification performance of a
> fine-tuned vision language model differ between microscopic and ordinary
> smartphone images for asbestos detection and asbestos type identification?

| Task | Target | Answers |
|---|---|---|
| `binary` | asbestos present vs absent | `Yes` / `No` |
| `type` | material category | 7 classes, below (configurable) |

| Code | Answer the model gives | Binary |
|---|---|---|
| A-AM | Amosite | asbestos |
| A-CF | Chrysotile Fluffy | asbestos |
| A-COF | Chrysotile and Organic Fiber | asbestos |
| A-CP | Chrysotile Pointy | asbestos |
| A-CRO | Crocidolite | asbestos |
| NA-CS | Cement Sheet | non-asbestos |
| NA-OF | Organic Fiber | non-asbestos |

Data folders and annotations use the codes; the model is trained to answer
with the descriptive names. `configs/classes_codes.yaml` trains on the bare
codes instead (an ablation).

## Repository layout

```
asbestos-vlm-hpc/
├── configs/
│   ├── classes_default.yaml          # classes, answer names, aliases (merging), drops
│   └── classes_codes.yaml            # same classes, model answers with bare codes
├── data/README.md                    # raw data layout + preparation
├── src/
│   ├── labels.py        # LabelSpace: rename / merge / drop classes
│   ├── boxes.py         # bounding-box geometry, box-preserving tiling
│   ├── manifest.py      # manifest (JSONL) I/O
│   ├── prepare_data.py  # raw images -> manifest with specimen-grouped splits
│   ├── augment.py       # offline augmentation that keeps boxes correct
│   ├── dataset.py       # PyTorch dataset, prompts, tiling, collator
│   ├── model.py         # Qwen2.5-VL + bitsandbytes 4/8-bit + LoRA
│   ├── train.py         # Hugging Face Trainer, resumable
│   └── evaluate.py      # metrics, confusion matrices, McNemar comparisons
├── slurm/
│   ├── common.sh        # shared paths/modules/venv (edit once)
│   ├── augment.sh       # CPU job: offline augmentation
│   ├── train_micro.sh   # GPU job: microscopic model
│   ├── train_phone.sh   # GPU job: smartphone model
│   ├── train_unified.sh # GPU job: optional model trained on both
│   └── evaluate.sh      # GPU job: cross-domain matrix + statistics
├── tests/               # CPU unit tests (labels, boxes, augmentation, stats)
├── requirements.txt
└── README.md
```

## Experimental design

Two models are trained with identical hyper-parameters: **micro** (microscope
images, tiled) and **phone** (smartphone photos). `slurm/evaluate.sh` then
evaluates every model on every test set:

| | microscopic test | smartphone test |
|---|---|---|
| **micro model** | in-domain | cross-domain |
| **phone model** | cross-domain | in-domain |
| *(unified model)* | in-domain | in-domain |
| *(zero-shot base)* | baseline | baseline |

and runs these comparisons (`results/comparisons/*.json`):

* `micro_vs_phone` – **the main research question**: in-domain microscopic
  vs in-domain smartphone performance.
* `unified_micro_vs_phone` – one model, two image types: separates the
  effect of the *image modality* from the effect of having a different model.
* `microtest_…` / `phonetest_…` – cross-domain transfer.
* `*_vs_zeroshot` – gain from fine-tuning.

**Statistics.** McNemar's test needs paired outcomes on the *same items*.
Microscope and phone images are different images, so they are paired by
**specimen** (`sample_id`): when the same physical sample was imaged both
ways, predictions are collapsed to one per specimen and McNemar's test is run
on the specimen-level correctness (exact binomial if < 25 discordant pairs,
otherwise χ² with continuity correction). Comparisons of two models on the
same test images are always paired. When the two sets share no specimens,
the script reports Fisher's exact test instead and says so. All comparisons
include bootstrap 95 % CIs for the accuracy and macro-F1 differences, and
Holm correction across the two tasks. Per-run metrics use a *cluster*
bootstrap over specimens, so several images of one sample are not treated
as independent evidence.

**Prediction.** Instead of parsing free text, the default `--scoring full`
computes the model's log-likelihood of each permitted answer and normalises
over classes. Every image gets a valid prediction and a probability vector
(→ ROC-AUC, PR-AUC, tile aggregation). Microscope tiles are aggregated to
one prediction per image: *P(asbestos) = max over tiles* (one fibre is enough),
and the type is the asbestos class with the most probability mass across
tiles. `--scoring generate` is available for free-text baselines.

## 1. Environment setup (once, on a login node)

```bash
# Where everything big lives. On OzSTAR/NT this is normally /fred/<project>/$USER.
export SCRATCH_BASE=/scratch/$USER          # or: /fred/oz123/$USER
export PROJECT_DIR=$SCRATCH_BASE/asbestos-vlm
mkdir -p $PROJECT_DIR && cd $PROJECT_DIR
git clone <your-repo-url> asbestos-vlm-hpc
cd asbestos-vlm-hpc
mkdir -p logs

module purge
module load python/3.10 cuda/12.1           # confirm names: module spider python ; module spider cuda

python -m venv $PROJECT_DIR/venv
source $PROJECT_DIR/venv/bin/activate
export PIP_CACHE_DIR=$PROJECT_DIR/pip_cache
pip install --upgrade pip wheel
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If you used a different `SCRATCH_BASE`, either edit the default at the top of
`slurm/common.sh` or put `export SCRATCH_BASE=...` in your `~/.bashrc`
(sbatch exports your environment to the job). If your site's module names
differ, set `MODULES="gcc/12.2.0 python/3.10.8 cuda/12.1.1"` (for example) the
same way.

**Download the model on the login node** (compute nodes usually have no
internet; jobs run with `HF_HUB_OFFLINE=1`):

```bash
export HF_HOME=$PROJECT_DIR/hf_cache
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

Run the CPU unit tests:

```bash
python -m pytest -q tests
```

## 2. Data preparation

See [data/README.md](data/README.md) for the folder layout and annotation
format. In short:

```bash
export DATA_DIR=$PROJECT_DIR/data
python -m src.prepare_data --data_dir $DATA_DIR --sample_id_regex '^(S\d+)' \
    --yolo_names $DATA_DIR/names.txt
sbatch slurm/augment.sh                     # or run src.augment interactively
```

### Changing and combining classes

All label handling goes through `configs/classes_default.yaml`:

```yaml
classes:
  a_am:  {name: Amosite,       asbestos: true}    # key: {name = text the model answers, binary target}
  ...
  na_of: {name: Organic Fiber, asbestos: false}
aliases: {}                               # e.g. old_folder_name: a_cf  (many-to-one = merge)
drop: [unknown]
```

Class keys are the codes in lower case with `_` (`A-COF` → `a_cof`), and
folder/annotation labels are matched case-insensitively with `-`, `_` and
spaces treated alike, so folders named `A-COF/` need no aliases. The answer
names are accepted as labels too (`Chrysotile Pointy/` → `a_cp`).

Or from the command line of `prepare_data`, `augment`, `train` and
`evaluate predict`:

```bash
--merge a_cf,a_cp=a_c:"Chrysotile"        # combine fluffy + pointy chrysotile (key:answer text)
--rename a_am="Amosite (brown asbestos)"   # change the answer text
--drop_class na_of                         # remove a class and its images
--class_config configs/classes_codes.yaml  # answer with the bare codes
```

The answer text is what the model learns to output. Descriptive names let
the model use what it already knows about these materials, while bare codes
must be learned from your data alone, so `classes_codes.yaml` is a
worthwhile ablation. Comparisons between runs should use the same label space.

Merging asbestos with non-asbestos classes is refused (the binary target
would be undefined). Merges and renames are applied to box labels too. The
final mapping is saved as `runs/<model>/label_space.json` and is reused
automatically during evaluation, so a model is always scored against the
classes it was trained on.

Pass training options through `sbatch`:
`sbatch slurm/train_micro.sh --merge a_cf,a_cp=a_c:Chrysotile` (with
`OUT=$RUNS_DIR/micro_merged` exported to keep runs apart).

`--scoring first_token` does not work with these classes (several answers
start with the same token, e.g. the three chrysotile classes); the default
`--scoring full` does.

### Making more data while keeping boxes correct

`src/augment.py` writes N augmented copies per training image with
albumentations, transforming image and Pascal VOC boxes together:

* **Microscopic:** flips and 90° rotations (exact for boxes), mild affine
  (rotation uses the `ellipse` box method, which keeps boxes around thin,
  rotated fibres tight), brightness/contrast, CLAHE, blur, noise. **Hue is
  never shifted**, because polarised-light and dispersion-staining colours
  identify the fibre type.
* **Smartphone:** flips, affine, perspective, lighting/colour/gamma, motion
  blur, JPEG compression.
* Boxes with less than `--min_visibility` of their area left are removed.
  `--require_boxes` throws away a copy if every box was lost.
* `--balance` gives minority classes more copies (per modality).
* Copies keep their source's `sample_id` and `split`. Only `train` is
  augmented, so there is no leakage into val/test, and evaluation ignores
  augmented records.
* `--preview N` writes box overlays to `augmented/_preview/` for checking by eye.

Microscope **tiling** (`--tile_size 512`) also adds data: each tile is its own
example, and with `--tile_label_source boxes` each tile is labelled from the
fibres inside it. Tiles from positive images that contain no annotated fibre
are dropped by default (`--empty_tile_policy drop`), because their true label
is uncertain.

## 3. Training

```bash
cd $PROJECT_DIR/asbestos-vlm-hpc
sbatch --account=oz123 slurm/train_micro.sh
sbatch --account=oz123 slurm/train_phone.sh
sbatch --account=oz123 slurm/train_unified.sh     # optional
squeue -u $USER ; tail -f logs/asb-train-micro-<jobid>.out
```

Any extra arguments go to `src/train.py` (`python -m src.train -h` lists them all):

```bash
sbatch slurm/train_micro.sh --epochs 5 --learning_rate 5e-5 --tune_vision --lora_r 32
```

* Adapters, processor, `label_space.json` and `train_config.json` are written to
  `$RUNS_DIR/<micro|phone|unified>`.
* Jobs resume from the latest checkpoint when resubmitted (`--resume auto`),
  so a 12 h walltime limit is not a problem.
* Memory: 4-bit 7B + bf16 vision tower with batch size 2 and gradient
  checkpointing uses about 20–30 GB of GPU memory. Lower `--max_pixels` or
  `--batch_size` if you run out; raise `--grad_accum` to keep the effective batch.
* `--tune_vision` also adds LoRA to the vision encoder. This is worth an
  ablation for microscope images, which are far from the pre-training data.
* Interactive debugging: `sinteractive`/`salloc --gres=gpu:1 ...`, then
  `source slurm/common.sh` and run `python -m src.train ... --max_train_samples 32 --epochs 1`.

## 4. Evaluation

```bash
sbatch --dependency=afterok:<micro_jobid>:<phone_jobid> slurm/evaluate.sh
# with the unified model and zero-shot baselines:
sbatch --export=ALL,MODELS="micro phone unified",ZERO_SHOT=1 slurm/evaluate.sh
```

Per run (`$RESULTS_DIR/<model>_on_<testset>/`):

| File | Contents |
|---|---|
| `predictions.csv` | one row per image and task: truth, prediction, class probabilities, `sample_id` |
| `tiles.csv` | tile-level probabilities (microscopic) |
| `metrics.json` | accuracy, balanced accuracy, macro/weighted precision/recall/F1, Cohen's κ, MCC, sensitivity/specificity, ROC-AUC, PR-AUC, per-class TP/FP/FN/TN, cluster-bootstrap 95 % CIs |
| `confusion_<task>.png/.csv` | confusion matrix (counts and row-normalised) |
| `per_class_<task>.csv` | per-class (one-vs-rest) precision/recall/F1 and confusion counts |

Plus `$RESULTS_DIR/comparisons/*.json` (McNemar / Fisher, CIs, Holm) and
`$RESULTS_DIR/summary.csv`.

Manual use:

```bash
python -m src.evaluate predict --adapter_dir $RUNS_DIR/micro --data_dir $DATA_DIR \
    --split test --modality smartphone --output_dir $RESULTS_DIR/micro_on_smartphone
python -m src.evaluate compare --a A/predictions.csv --b B/predictions.csv \
    --name_a micro --name_b phone --task both --output cmp.json
```

## Notes and caveats

* **Confounds.** The micro and phone models differ in data size and class
  balance as well as modality. Report class counts per modality, and use the
  unified model and the specimen-paired comparison to support claims about
  the modality itself.
* **Prompts mention the image type** (`microscope image` / `smartphone
  photograph`). Use `--no_modality_hint` for an ablation.
* **Threshold.** Binary decisions use P(asbestos) ≥ 0.5 by default
  (`--threshold`). For a safety screening tool you may prefer a threshold
  chosen on the validation set for high sensitivity. Report how it was chosen.
* **Cluster specifics.** The `#SBATCH --account`, partition, module names and
  storage paths follow the conventions you gave. Check them against the
  current Swinburne HPC documentation (for example, project storage on
  OzSTAR is `/fred/<project>`, and node-local disk is requested with `--tmp`
  and used through `$JOBFS`; set `STAGE_TO_JOBFS=1` to copy data there).
