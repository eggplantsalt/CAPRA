#!/usr/bin/env bash
# eval_capra.sh
# Evaluate a CAPRA-trained checkpoint on LIBERO / SafeLIBERO.
#
# Usage:
#   bash scripts/capra/eval_capra.sh [CHECKPOINT] [TASK_SUITE] [UNNORM_KEY]

set -euo pipefail

CHECKPOINT="${1:-tmp/models/openvla-oft-libero-capra}"
TASK_SUITE="${2:-libero_spatial}"
UNNORM_KEY="${3:-libero_spatial_no_noops}"

echo "[eval_capra] checkpoint  : $CHECKPOINT"
echo "[eval_capra] task_suite  : $TASK_SUITE"
echo "[eval_capra] unnorm_key  : $UNNORM_KEY"

python -m experiments.robot.capra.run_capra_eval \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name       "$TASK_SUITE" \
    --unnorm_key            "$UNNORM_KEY" \
    --num_trials_per_task   50 \
    --use_l1_regression     True \
    --use_proprio           True \
    --num_images_in_input   2 \
    --center_crop           True \
    --local_log_dir         experiments/logs/capra \
    --seed                  7
