#!/bin/bash
# Shared job environment. Sourced by every slurm/*.sh script - do not sbatch this file.
# Every variable can be overridden at submission time, e.g.
#   sbatch --export=ALL,SCRATCH_BASE=/fred/oz123/$USER slurm/train_micro.sh

# ---------------------------------------------------------------- paths
# High-speed project storage. On OzSTAR / Ngarrgu Tindebeek this is usually
# /fred/<project>/$USER; set SCRATCH_BASE accordingly (never use $HOME: small quota).
: "${SCRATCH_BASE:=/scratch/${USER}}"
: "${PROJECT_DIR:=${SCRATCH_BASE}/asbestos-vlm}"
: "${DATA_DIR:=${PROJECT_DIR}/data}"
: "${RUNS_DIR:=${PROJECT_DIR}/runs}"
: "${RESULTS_DIR:=${PROJECT_DIR}/results}"
: "${VENV_DIR:=${PROJECT_DIR}/venv}"
: "${MANIFEST:=${DATA_DIR}/manifest_aug.jsonl}"      # falls back to manifest.jsonl below
: "${MICRO_CLASS_CONFIG:=configs/classes_micro.yaml}"   # microscopic type classes
: "${MACRO_CLASS_CONFIG:=configs/classes_macro.yaml}"   # macro (smartphone) type classes
: "${MODEL_NAME:=Qwen/Qwen2.5-VL-7B-Instruct}"

# ---------------------------------------------------------------- modules
# Check names with `module spider python` / `module spider cuda` on the login node.
: "${MODULES:=python/3.10 cuda/12.1}"

# ---------------------------------------------------------------- caches
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_DIR}/pip_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/.mplconfig}"
# Compute nodes usually have no internet: download the model on the login node first (README).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- setup
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_DIR}"
if [[ ! -f src/train.py ]]; then
    echo "ERROR: submit jobs from the repository root (REPO_DIR=${REPO_DIR})" >&2
    exit 1
fi
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

module purge
# shellcheck disable=SC2086
module load ${MODULES}

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: virtual environment not found at ${VENV_DIR}. See README 'Environment setup'." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "NOTE: ${MANIFEST} not found, using ${DATA_DIR}/manifest.jsonl"
    MANIFEST="${DATA_DIR}/manifest.jsonl"
fi
mkdir -p "${RUNS_DIR}" "${RESULTS_DIR}" "${MPLCONFIGDIR}"
export PROJECT_DIR DATA_DIR RUNS_DIR RESULTS_DIR MANIFEST MICRO_CLASS_CONFIG MACRO_CLASS_CONFIG MODEL_NAME

# Optional: copy the dataset to node-local disk for faster I/O.
# Request local disk at submission (e.g. sbatch --tmp=100G) and set STAGE_TO_JOBFS=1.
if [[ "${STAGE_TO_JOBFS:-0}" == "1" ]]; then
    LOCAL="${JOBFS:-${TMPDIR:-/tmp}}/asbestos-data"
    echo "Staging ${DATA_DIR} -> ${LOCAL}"
    mkdir -p "${LOCAL}"
    rsync -a --exclude '_preview' "${DATA_DIR}/" "${LOCAL}/"
    MANIFEST="${LOCAL}/$(basename "${MANIFEST}")"
    DATA_DIR="${LOCAL}"
    export MANIFEST DATA_DIR
fi

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
echo "REPO_DIR=${REPO_DIR}  DATA_DIR=${DATA_DIR}  MANIFEST=${MANIFEST}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| peft', peft.__version__)"
