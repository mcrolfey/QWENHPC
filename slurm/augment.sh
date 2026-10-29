#!/bin/bash
#SBATCH --job-name=asb-augment
#SBATCH --account=oz000                 # CHANGE to your project, or override: sbatch --account=ozXXX
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# CPU-only offline augmentation (no GPU needed; uses the default partition).
# Writes $DATA_DIR/manifest_aug.jsonl and images under $DATA_DIR/augmented/.
#   sbatch slurm/augment.sh --copies 4 --balance --preview 30
set -euo pipefail
export STAGE_TO_JOBFS=0
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/common.sh"

python -m src.augment \
    --data_dir "${DATA_DIR}" \
    --manifest "${DATA_DIR}/manifest.jsonl" \
    --output manifest_aug.jsonl \
    --class_config "${CLASS_CONFIG}" \
    --copies 3 \
    --balance \
    --require_boxes \
    --preview 20 \
    "$@"
