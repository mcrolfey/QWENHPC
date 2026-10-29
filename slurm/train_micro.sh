#!/bin/bash
#SBATCH --job-name=asb-train-micro
#SBATCH --account=oz000                 # CHANGE to your project, or override: sbatch --account=ozXXX
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#
# Fine-tune on MICROSCOPIC images (512x512 tiles labelled from fibre boxes).
# Extra arguments are passed through to src.train, e.g.
#   sbatch slurm/train_micro.sh --epochs 5 --tune_vision
# Re-submitting the same job resumes from the latest checkpoint.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/common.sh"

OUT="${OUT:-${RUNS_DIR}/micro}"

srun python -m src.train \
    --modality microscopic \
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
    --empty_tile_policy drop \
    --online_aug \
    --balance \
    --max_pixels $((512 * 512)) \
    "$@"

echo "Done: adapter in ${OUT}"
