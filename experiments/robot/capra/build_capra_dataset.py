# ===== 向后兼容导入 shim =====
# 本文件是一个转发层（shim），实际代码已迁移到子包中。
# 保留本文件是为了让旧的导入语句继续工作，例如：
#   from experiments.robot.capra.build_capra_dataset import ...  ← 仍然有效
#
# 功能说明：安全目标分布构建器，计算 q_hat_t 并输出训练样本
#
# 实际实现位置：experiments.robot.capra.scene.build_capra_dataset
# 如需查看或修改代码，请打开上方的实际实现文件。
# ================================
# ruff: noqa
from experiments.robot.capra.scene.build_capra_dataset import *  # noqa: F401,F403
