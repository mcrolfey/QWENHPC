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
# Results go to $RESULTS_DIR/<model>_on_<testset>/: predictions.csv / metrics.json in the
# model's original classes, and predictions_shared.csv / metrics_shared.json with the type
# task mapped onto the shared classes of $CLASS_MAPPING (comparable across datasets).
# Existing results are skipped; set FORCE=1 to recompute. CLASS_MAPPING="" disables mapping.
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
CLASS_MAPPING="${CLASS_MAPPING-configs/class_mapping_shared.yaml}"
MAPPING_ARGS=()
if [[ -n "${CLASS_MAPPING}" ]]; then MAPPING_ARGS=(--class_mapping "${CLASS_MAPPING}"); fi

testset_args() {   # preprocessing per test set; keep in sync with train_*.sh
    if [[ "$1" == "microscopic" ]]; then
        echo "--tile_size 512 --tile_overlap 0.1 --max_pixels $((512 * 512))"
    else
        echo "--tile_size 0 --max_pixels $((1024 * 28 * 28))"
    fi
}

testset_config() {   # class config of each dataset (micro and macro have different type classes)
    if [[ "$1" == "microscopic" ]]; then echo "${MICRO_CLASS_CONFIG}"; else echo "${MACRO_CLASS_CONFIG}"; fi
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
        ${MAPPING_ARGS[@]+"${MAPPING_ARGS[@]}"} \
        "$@"
}

for model in ${MODELS}; do
    if [[ ! -f "${RUNS_DIR}/${model}/adapter_config.json" ]]; then
        echo "WARNING: no adapter in ${RUNS_DIR}/${model}, skipping"; continue
    fi
    for testset in ${TESTSETS}; do
        # Each test set is presented identically to every model (same tiling and
        # visual-token budget), so differences come from the model, not preprocessing.
        # Cross-domain runs (e.g. micro model on macro images) score the type task in the
        # shared classes only, because the two datasets have different original classes.
        predict "${model}" "${RUNS_DIR}/${model}" "${testset}" $(testset_args "${testset}")
    done
done

if [[ "${ZERO_SHOT}" == "1" ]]; then
    for testset in ${TESTSETS}; do
        predict "zeroshot" none "${testset}" --class_config "$(testset_config "${testset}")" \
            --base_model "${MODEL_NAME}" --task both $(testset_args "${testset}")
    done
fi

# ------------------------------------------------------------ comparisons
compare_files() {   # compare_files <file> <a> <b> <out-name> <name_a> <name_b>
    local a="${RESULTS_DIR}/$2/$1" b="${RESULTS_DIR}/$3/$1"
    if [[ -f "${a}" && -f "${b}" ]]; then
        python -m src.evaluate compare --a "${a}" --b "${b}" --name_a "$5" --name_b "$6" \
            --task both --output "${RESULTS_DIR}/comparisons/$4.json"
    else
        echo "skip comparison $4 (missing $1)"
    fi
}
compare() {   # compare <a> <b> <out-name> <name_a> <name_b>: original and shared classes
    compare_files predictions.csv "$1" "$2" "$3" "$4" "$5"
    compare_files predictions_shared.csv "$1" "$2" "$3_shared" "$4" "$5"
}
mkdir -p "${RESULTS_DIR}/comparisons"
# <name>.json compares the original classes: tasks whose class sets differ between A and
# B (the type task of micro vs macro) are skipped and listed under "skipped".
# <name>_shared.json compares the shared classes, so the type task is included.
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
for f in sorted(glob.glob(f"{res}/*_on_*/metrics*.json")):
    m = json.load(open(f))
    classes = "shared" if f.endswith("_shared.json") else "original"
    for task, t in m["tasks"].items():
        if classes == "shared" and task == "binary":
            continue  # identical to the original binary result
        rows.append({"run": os.path.basename(os.path.dirname(f)), "classes": classes, "task": task, "n": t["n"],
                     "accuracy": t["accuracy"], "balanced_acc": t["balanced_accuracy"],
                     "f1_macro": t["f1_macro"], "kappa": t["cohen_kappa"],
                     "acc_ci95": t.get("ci95", {}).get("accuracy")})
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(f"{res}/summary.csv", index=False)
    print(df.to_string(index=False))
EOF
echo "Summary: ${RESULTS_DIR}/summary.csv"
