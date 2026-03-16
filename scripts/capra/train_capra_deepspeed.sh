#!/usr/bin/env bash
# =============================================================
# train_capra_deepspeed.sh -- CAPRA 微调训练（DeepSpeed 版）
# =============================================================
#
# 用法
# ----
#   bash scripts/capra/train_capra_deepspeed.sh                          # 默认参数
#   bash scripts/capra/train_capra_deepspeed.sh CKPT DATASET CACHE_ROOT  # 自定义参数
#   bash scripts/capra/train_capra_deepspeed.sh CKPT DATASET CACHE_ROOT NUM_GPUS  # 多卡
#
# 位置参数（均可选）
#   $1  VLA 检查点路径  (默认: tmp/models/openvla-oft-libero)
#   $2  数据集名称      (默认: libero_spatial_no_noops)
#   $3  挖掘缓存根目录  (默认: tmp/capra_cache)
#   $4  GPU 数量        (默认: 1)
#   $5  DeepSpeed 配置  (默认: vla-scripts/ds_config_zero2.json)
#
# ⚠️  shuffle_buffer_size 必须 <= 2000，代码中有 assert 保护
# =============================================================

set -euo pipefail

CHECKPOINT="${1:-tmp/models/openvla-oft-libero}"
DATASET="${2:-libero_spatial_no_noops}"
CACHE_ROOT="${3:-tmp/capra_cache}"
NUM_GPUS="${4:-1}"
DS_CONFIG="${5:-vla-scripts/ds_config_zero2.json}"

echo "[train_capra_deepspeed] ================================================"
echo "[train_capra_deepspeed] checkpoint   : $CHECKPOINT"
echo "[train_capra_deepspeed] dataset      : $DATASET"
echo "[train_capra_deepspeed] cache_root   : $CACHE_ROOT"
echo "[train_capra_deepspeed] num_gpus     : $NUM_GPUS"
echo "[train_capra_deepspeed] ds_config    : $DS_CONFIG"
echo "[train_capra_deepspeed] mode         : CAPRA (capra_enabled=True)"
echo "[train_capra_deepspeed] ================================================"

deepspeed --num_gpus "$NUM_GPUS" \
    vla-scripts/finetune_capra_deepspeed.py \
    --vla_path              "$CHECKPOINT" \
    --dataset_name          "$DATASET" \
    --data_root_dir         tmp/datasets/rlds \
    --cache_root            "$CACHE_ROOT" \
    --run_root_dir          runs \
    --deepspeed_config      "$DS_CONFIG" \
    --capra_enabled         True \
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
    --capra_warmup_steps    500 \
    --lam                   0.1 \
    --rho                   0.5 \
    --beta                  1.0 \
    --capra_gamma           1.0 \
    --image_aug             True

echo "[train_capra_deepspeed] Done. Checkpoints in: runs/*${DATASET}*capra*deepspeed*/"
