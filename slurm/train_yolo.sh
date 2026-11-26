#!/bin/bash
#SBATCH --job-name=asb-yolo
#SBATCH --account=oz000                 # CHANGE to your project, or override: sbatch --account=ozXXX
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#
# YOLO box detector for autolabelling the macro (smartphone) photos.
# 1. exports the manifest's macro boxes as a YOLO dataset (src.yolo_data),
# 2. trains and scores it on the test split (src.yolo_train).
# Run `python -m src.yolo_train --setup ...` on the login node first (README, "YOLO autolabelling").
# Extra arguments go to src.yolo_train, e.g.
#   sbatch slurm/train_yolo.sh --epochs 200 --imgsz 1280
# Environment overrides: YOLO_MANIFEST, YOLO_CLASS_MODE (type|binary|object), YOLO_MODEL, OUT.
# Re-submitting the same job resumes from weights/last.pt.
set -euo pipefail
export STAGE_TO_JOBFS=0
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/common.sh"

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${PROJECT_DIR}/.ultralytics}"
mkdir -p "${YOLO_CONFIG_DIR}"           # Ultralytics ignores the setting for a missing folder
export YOLO_OFFLINE=true
: "${YOLO_MANIFEST:=${DATA_DIR}/manifest_grouped.jsonl}"   # near-duplicate grouped splits
: "${YOLO_CLASS_MODE:=type}"
: "${YOLO_MODEL:=${PROJECT_DIR}/weights/yolo11m.pt}"
YOLO_DATA="${YOLO_DATA:-${DATA_DIR}/yolo_macro_${YOLO_CLASS_MODE}}"
OUT="${OUT:-${RUNS_DIR}/yolo_macro_${YOLO_CLASS_MODE}}"

python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

if [[ ! -f "${OUT}/weights/last.pt" ]]; then
    python -m src.yolo_data \
        --data_dir "${DATA_DIR}" \
        --manifest "${YOLO_MANIFEST}" \
        --modality smartphone \
        --class_config "${MACRO_CLASS_CONFIG}" \
        --class_mode "${YOLO_CLASS_MODE}" \
        --out_dir "${YOLO_DATA}"
fi

srun python -m src.yolo_train \
    --data "${YOLO_DATA}/data.yaml" \
    --model "${YOLO_MODEL}" \
    --run_dir "${OUT}" \
    --epochs 150 \
    --patience 40 \
    --imgsz 1024 \
    --batch 16 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"

echo "Done: ${OUT}/weights/best.pt, test scores in ${OUT}/test_metrics.json"
