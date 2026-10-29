#!/bin/bash
#SBATCH --job-name=asb-train-phone
#SBATCH --account=oz000                 # CHANGE to your project, or override: sbatch --account=ozXXX
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#
# Fine-tune on SMARTPHONE images (whole image, up to ~1024 visual tokens).
# Extra arguments are passed through to src.train, e.g.
#   sbatch slurm/train_phone.sh --epochs 5 --learning_rate 5e-5
# Re-submitting the same job resumes from the latest checkpoint.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/common.sh"

OUT="${OUT:-${RUNS_DIR}/phone}"

srun python -m src.train \
    --modality smartphone \
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
    --tile_size 0 \
    --online_aug \
    --balance \
    --max_pixels $((1024 * 28 * 28)) \
    "$@"

echo "Done: adapter in ${OUT}"
