#!/usr/bin/env bash
# =============================================================
# mine_capra.sh -- CAPRA 离线挖掘监督信号
# =============================================================
#
# 作用
# ----
# 使用已有的 VLA 基线检查点，在 LIBERO 环境中运行离线挖掘。
# 对每个 episode 的每个时间步：
#   1. 保存 MuJoCo 快照
#   2. 采样 K 个候选动作（nominal + 高斯噪声扰动）
#   3. 对每个候选执行 H_s 步短时 rollout
#   4. 计算任务进度 P_t 和足迹 F_t
#   5. 过滤等价集 E_t，计算安全目标分布 q_hat_t
#   6. 将监督信号存入 {cache_root}/{dataset_name}/episode_XXXX.npz
#
# 断点续传：已存在的 episode 缓存文件自动跳过，中途中断可直接重启
#
# 前提条件
# --------
#   1. conda activate openvla
#   2. 检查点存在于 tmp/models/openvla-oft-libero/
#   3. pip install libero（LIBERO 仿真环境）
#
# 用法
# ----
#   bash scripts/capra/mine_capra.sh                      # 使用默认参数
#   bash scripts/capra/mine_capra.sh CKPT DATASET CACHE   # 自定义参数
#
# 位置参数（均可选）
#   $1  VLA 检查点路径  (默认: tmp/models/openvla-oft-libero)
#   $2  数据集名称      (默认: libero_spatial)
#   $3  缓存根目录      (默认: tmp/capra_cache)
#
# 快速验证（只挖 1 个 episode，几分钟内完成）
#   python -m experiments.robot.capra.mining.run_capra_mining \
#       --pretrained_checkpoint tmp/models/openvla-oft-libero \
#       --num_mining_episodes 1
# =============================================================

set -euo pipefail

# -------------------------------------------------------
# 参数读取（位置参数覆盖默认值）
# -------------------------------------------------------
CHECKPOINT="${1:-tmp/models/openvla-oft-libero}"  # VLA 检查点路径
DATASET="${2:-libero_spatial}"                     # LIBERO 任务套件名称
CACHE_ROOT="${3:-tmp/capra_cache}"                 # 挖掘缓存输出目录

echo "[mine_capra] ================================================"
echo "[mine_capra] checkpoint : $CHECKPOINT"
echo "[mine_capra] dataset    : $DATASET"
echo "[mine_capra] cache_root : $CACHE_ROOT"
echo "[mine_capra] ================================================"

# -------------------------------------------------------
# 主挖掘命令
# -------------------------------------------------------
# 参数说明：
#   --num_mining_episodes  挖掘的 episode 数量（50 个约需 2-4 小时）
#   --K                    每步候选动作数量（越多质量越好，速度越慢）
#   --H_s                  每次候选 rollout 执行的步数
#   --W                    前驱归因回看窗口大小
#   --seed                 随机种子（固定以保证可复现）
python -m experiments.robot.capra.mining.run_capra_mining \
    --pretrained_checkpoint "$CHECKPOINT" \
    --dataset_name          "$DATASET" \
    --cache_root            "$CACHE_ROOT" \
    --num_mining_episodes   50 \
    --K                     8 \
    --H_s                   5 \
    --W                     10 \
    --seed                  7

echo "[mine_capra] 挖掘完成。缓存文件在: $CACHE_ROOT/$DATASET/"
