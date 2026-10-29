#!/bin/bash
#SBATCH --job-name=asb-train-unified
#SBATCH --account=oz000                 # CHANGE to your project, or override: sbatch --account=ozXXX
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#
# Optional: one model trained on BOTH modalities. Evaluating it separately on
# microscopic and smartphone test images isolates the effect of image
# modality from the effect of training data (see README "Experimental design").
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/common.sh"

OUT="${OUT:-${RUNS_DIR}/unified}"

srun python -m src.train \
    --modality unified \
    --data_dir "${DATA_DIR}" \
    --manifest "${MANIFEST}" \
    --class_config "${CLASS_CONFIG}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUT}" \
    --task both \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --learning_rate 1e-4 \
    --tile_size 512 \
    --tile_overlap 0.1 \
    --max_tiles_per_image 16 \
    --tile_label_source boxes \
    --online_aug \
    --balance \
    --max_pixels $((1024 * 28 * 28)) \
    "$@"

echo "Done: adapter in ${OUT}"
