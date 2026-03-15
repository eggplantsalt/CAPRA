#!/usr/bin/env bash
# train_capra.sh
# CAPRA fine-tuning on top of a pre-mined supervision cache.
#
# Usage:
#   bash scripts/capra/train_capra.sh [CHECKPOINT] [DATASET] [CACHE_ROOT]

set -euo pipefail

CHECKPOINT="${1:-tmp/models/openvla-oft-libero}"
DATASET="${2:-libero_spatial}"
CACHE_ROOT="${3:-tmp/capra_cache}"

echo "[train_capra] checkpoint : $CHECKPOINT"
echo "[train_capra] dataset    : $DATASET"
echo "[train_capra] cache_root : $CACHE_ROOT"

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune_capra.py \
    --vla_path              "$CHECKPOINT" \
    --dataset_name          "$DATASET" \
    --data_root_dir         tmp/datasets/rlds \
    --cache_root            "$CACHE_ROOT" \
    --run_root_dir          runs \
    --use_l1_regression     True \
    --use_diffusion         False \
    --use_proprio           True \
    --num_images_in_input   2 \
    --use_lora              True \
    --lora_rank             32 \
    --batch_size            8 \
    --shuffle_buffer_size   2000 \
    --learning_rate         5e-4 \
    --max_steps             200000 \
    --save_freq             10000 \
    --lam                   0.1 \
    --beta                  1.0 \
    --image_aug             True
