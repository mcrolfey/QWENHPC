#!/bin/bash
#SBATCH --job-name=asb-evaluate
#SBATCH --account=oz000                 # CHANGE to your project, or override: sbatch --account=ozXXX
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#
# Cross-domain evaluation matrix + statistical comparisons.
#
#   models (adapters in $RUNS_DIR):   micro, phone [, unified] [, zero-shot]
#   test sets:                        microscopic, smartphone
#
# Results go to $RESULTS_DIR/<model>_on_<testset>/ (predictions.csv, metrics.json,
# confusion_*.png). Existing results are skipped; set FORCE=1 to recompute.
#
#   sbatch slurm/evaluate.sh
#   sbatch --export=ALL,MODELS="micro phone unified",ZERO_SHOT=1 slurm/evaluate.sh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/common.sh"

MODELS="${MODELS:-micro phone}"
TESTSETS="${TESTSETS:-microscopic smartphone}"
SPLIT="${SPLIT:-test}"
SCORING="${SCORING:-full}"
ZERO_SHOT="${ZERO_SHOT:-0}"
FORCE="${FORCE:-0}"

testset_args() {   # preprocessing per test set; keep in sync with train_*.sh
    if [[ "$1" == "microscopic" ]]; then
        echo "--tile_size 512 --tile_overlap 0.1 --max_pixels $((512 * 512))"
    else
        echo "--tile_size 0 --max_pixels $((1024 * 28 * 28))"
    fi
}

predict() {   # predict <name> <adapter_dir|none> <testset> [extra args...]
    local name="$1" adapter="$2" testset="$3"; shift 3
    local out="${RESULTS_DIR}/${name}_on_${testset}"
    if [[ -f "${out}/predictions.csv" && "${FORCE}" != "1" ]]; then
        echo "skip ${out} (exists)"; return
    fi
    echo ">>> ${name} on ${testset}"
    srun python -m src.evaluate predict \
        --adapter_dir "${adapter}" \
        --data_dir "${DATA_DIR}" \
        --manifest "${MANIFEST}" \
        --split "${SPLIT}" \
        --modality "${testset}" \
        --scoring "${SCORING}" \
        --num_workers $(( ${SLURM_CPUS_PER_TASK:-8} - 1 )) \
        --output_dir "${out}" \
        "$@"
}

for model in ${MODELS}; do
    if [[ ! -f "${RUNS_DIR}/${model}/adapter_config.json" ]]; then
        echo "WARNING: no adapter in ${RUNS_DIR}/${model}, skipping"; continue
    fi
    for testset in ${TESTSETS}; do
        # Each test set is presented identically to every model (same tiling and
        # visual-token budget), so differences come from the model, not preprocessing.
        predict "${model}" "${RUNS_DIR}/${model}" "${testset}" $(testset_args "${testset}")
    done
done

if [[ "${ZERO_SHOT}" == "1" ]]; then
    for testset in ${TESTSETS}; do
        predict "zeroshot" none "${testset}" --class_config "${CLASS_CONFIG}" \
            --base_model "${MODEL_NAME}" --task both $(testset_args "${testset}")
    done
fi

# ------------------------------------------------------------ comparisons
compare() {   # compare <a> <b> <out-name> <name_a> <name_b>
    local a="${RESULTS_DIR}/$1/predictions.csv" b="${RESULTS_DIR}/$2/predictions.csv"
    if [[ -f "${a}" && -f "${b}" ]]; then
        python -m src.evaluate compare --a "${a}" --b "${b}" --name_a "$4" --name_b "$5" \
            --task both --output "${RESULTS_DIR}/comparisons/$3.json"
    else
        echo "skip comparison $3 (missing predictions)"
    fi
}
mkdir -p "${RESULTS_DIR}/comparisons"
# Main research question: microscopic vs smartphone (in-domain models).
compare micro_on_microscopic phone_on_smartphone micro_vs_phone microscopic smartphone
# Same unified model, microscopic vs smartphone images of the same specimens.
compare unified_on_microscopic unified_on_smartphone unified_micro_vs_phone microscopic smartphone
# Cross-domain transfer: in-domain vs out-of-domain on the same test images (always paired).
compare micro_on_microscopic phone_on_microscopic microtest_micro_vs_phone_model micro_model phone_model
compare phone_on_smartphone micro_on_smartphone phonetest_phone_vs_micro_model phone_model micro_model
# Fine-tuning gain over zero-shot.
compare micro_on_microscopic zeroshot_on_microscopic micro_vs_zeroshot finetuned zeroshot
compare phone_on_smartphone zeroshot_on_smartphone phone_vs_zeroshot finetuned zeroshot

python - <<'EOF'
import json, os, glob
import pandas as pd
res = os.environ["RESULTS_DIR"]
rows = []
for f in sorted(glob.glob(f"{res}/*_on_*/metrics.json")):
    m = json.load(open(f))
    for task, t in m["tasks"].items():
        rows.append({"run": os.path.basename(os.path.dirname(f)), "task": task, "n": t["n"],
                     "accuracy": t["accuracy"], "balanced_acc": t["balanced_accuracy"],
                     "f1_macro": t["f1_macro"], "kappa": t["cohen_kappa"],
                     "acc_ci95": t.get("ci95", {}).get("accuracy")})
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(f"{res}/summary.csv", index=False)
    print(df.to_string(index=False))
EOF
echo "Summary: ${RESULTS_DIR}/summary.csv"
