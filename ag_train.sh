#!/usr/bin/env bash
# Activate the COSORK environment before running this file.
# Default: 4-class and 5-class, substrate + tsfresh, best_quality, 3 hours each.
# Optional full feature/time comparisons: RUN_BENCHMARKS=1 bash ag_train.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
NUM_CPUS=${NUM_CPUS:-0}
NUM_GPUS=${NUM_GPUS:-0}

train_model() {
    "$PYTHON" -u ag_train.py \
        --class_num "$1" --feat "$2" --hours "$3" \
        --sp_mode s --ag_train_quality best_quality --random_seed 42 \
        --num_cpus "$NUM_CPUS" --num_gpus "$NUM_GPUS" \
        --verbose 2 --evaluate_on_test_data
}

for class_num in 4 5; do
    train_model "$class_num" tsfresh_raw 3.0
done

if [[ "${RUN_BENCHMARKS:-0}" == 1 ]]; then
    for class_num in 4 5; do
        train_model "$class_num" raw 3.0
        train_model "$class_num" tsfresh 3.0
        train_model "$class_num" tsfresh_raw 1.0
        train_model "$class_num" tsfresh_raw 2.0
    done
fi
