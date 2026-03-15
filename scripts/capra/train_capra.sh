#!/usr/bin/env bash
# =============================================================
# train_capra.sh -- CAPRA 微调训练
# =============================================================
#
# 作用
# ----
# 在已挖掘的监督信号缓存基础上，对 VLA 模型进行微调。
# 默认启用 CAPRA 模式（capra_enabled=True）：
#   总损失 = L_anchor（任务损失） + lambda * L_capra（安全 KL 损失）
#
# 两种模式
# --------
#   CAPRA 模式（默认）：需要先运行 mine_capra.sh 生成挖掘缓存
#   Baseline 模式：等价于原版 finetune.py，无 CAPRA 开销
#     bash scripts/capra/train_capra.sh  # 然后在末尾添加 --capra_enabled False
#
# 前提条件
# --------
#   1. conda activate openvla
#   2. mine_capra.sh 已运行（tmp/capra_cache/{dataset}/ 下有 .npz 文件）
#   3. RLDS 数据集存在于 tmp/datasets/rlds/{dataset}/
#   4. 检查点存在于 $CHECKPOINT
#
# 用法
# ----
#   bash scripts/capra/train_capra.sh                         # 默认参数
#   bash scripts/capra/train_capra.sh CKPT DATASET CACHE_ROOT # 自定义参数
#
# 位置参数（均可选）
#   $1  VLA 检查点路径  (默认: tmp/models/openvla-oft-libero)
#   $2  数据集名称      (默认: libero_spatial)
#   $3  挖掘缓存根目录  (默认: tmp/capra_cache)
#
# ⚠️  关键警告
# -----------
#   shuffle_buffer_size 必须 <= 2000
#   更大的值会导致 TensorFlow RLDS 加载器耗尽内存，进程被 kill
# =============================================================

set -euo pipefail

# -------------------------------------------------------
# 参数读取
# -------------------------------------------------------
CHECKPOINT="${1:-tmp/models/openvla-oft-libero}"  # 起始检查点
DATASET="${2:-libero_spatial}"                     # RLDS 数据集名称
CACHE_ROOT="${3:-tmp/capra_cache}"                 # 挖掘缓存根目录

echo "[train_capra] ================================================"
echo "[train_capra] checkpoint  : $CHECKPOINT"
echo "[train_capra] dataset     : $DATASET"
echo "[train_capra] cache_root  : $CACHE_ROOT"
echo "[train_capra] mode        : CAPRA (capra_enabled=True)"
echo "[train_capra] ================================================"

# -------------------------------------------------------
# 训练命令
# -------------------------------------------------------
# torchrun 参数：
#   --standalone         单机模式，不需要 master node
#   --nnodes 1           只有 1 台机器
#   --nproc-per-node 1   单卡（多卡改为 GPU 数量，如 4）
#
# 训练参数说明：
#   --capra_enabled True     启用 CAPRA 损失（False=纯 Baseline）
#   --lam 0.1               CAPRA 损失权重 lambda
#   --rho 0.5               前驱归因上权因子
#   --beta 1.0              安全目标分布温度
#   --capra_warmup_steps 500 前 500 步只跑 anchor loss，让模型先稳定
#   --capra_gamma 1.0       候选评分分布温度（0=均匀先验）
#   --shuffle_buffer_size 2000  ⚠️ 不要改这个值
#   --use_lora True         使用 LoRA 参数高效微调（减少显存需求）
#   --lora_rank 32          LoRA 秩（越大容量越强，但显存越多）
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune_capra.py \
    --vla_path              "$CHECKPOINT" \
    --dataset_name          "$DATASET" \
    --data_root_dir         tmp/datasets/rlds \
    --cache_root            "$CACHE_ROOT" \
    --run_root_dir          runs \
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

echo "[train_capra] 训练完成。检查点在: runs/CAPRA-${DATASET}-*/checkpoints/"
