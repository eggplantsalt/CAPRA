#!/usr/bin/env bash
# eval_capra.sh -- Evaluate a model on LIBERO with CAPRA metrics.
#
# Works for both baseline and CAPRA-trained models.
# CAPRA signal collection (footprint, displacement, topple, support-break)
# runs regardless -- it measures the *intrinsic* safety preference of the model.
#
# SPIR/EAR are 0 when --capra_eval_K 0 (obs-only, default).
# Set --capra_eval_K 8 to enable live counterfactual eval (requires MuJoCo
# sim.get_state/set_state -- slower but gives non-zero SPIR/EAR).
#
# Usage:
#   bash scripts/capra/eval_capra.sh                              # defaults
#   bash scripts/capra/eval_capra.sh tmp/models/my-capra-ckpt
#   bash scripts/capra/eval_capra.sh tmp/models/my-capra-ckpt libero_spatial 50 8
#
# Positional args (all optional):
#   $1  checkpoint path    (default: tmp/models/openvla-oft-libero)
#   $2  task suite         (default: libero_spatial)
#   $3  trials per task    (default: 50)
#   $4  capra_eval_K       (default: 0  -- obs-only; use 8 for live CF eval)

set -euo pipefail

CHECKPOINT="${1:-tmp/models/openvla-oft-libero}"
SUITE="${2:-libero_spatial}"
TRIALS="${3:-50}"
CF_K="${4:-0}"

echo "[eval_capra] checkpoint  : $CHECKPOINT"
echo "[eval_capra] suite       : $SUITE"
echo "[eval_capra] trials      : $TRIALS"
echo "[eval_capra] capra_eval_K: $CF_K"

python -m experiments.robot.capra.run_capra_eval \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name       "$SUITE" \
    --num_trials_per_task   "$TRIALS" \
    --use_l1_regression     True \
    --use_diffusion         False \
    --use_proprio           True \
    --num_images_in_input   2 \
    --center_crop           True \
    --num_open_loop_steps   8 \
    --lora_rank             32 \
    --env_img_res           256 \
    --capra_eval_K          "$CF_K" \
    --capra_eval_sigma      0.02 \
    --progress_floor        0.20 \
    --epsilon_p_abs         0.05 \
    --epsilon_p_rel         0.10 \
    --alpha_d               1.0 \
    --alpha_i               1.0 \
    --alpha_r               2.0 \
    --local_log_dir         experiments/logs/capra
