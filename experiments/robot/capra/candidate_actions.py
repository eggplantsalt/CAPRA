# ===== 向后兼容导入 shim =====
# 本文件是一个转发层（shim），实际代码已迁移到子包中。
# 保留本文件是为了让旧的导入语句继续工作，例如：
#   from experiments.robot.capra.candidate_actions import ...  ← 仍然有效
#
# 功能说明：候选动作采样，nominal 动作 + 高斯噪声扰动
#
# 实际实现位置：experiments.robot.capra.mining.candidate_actions
# 如需查看或修改代码，请打开上方的实际实现文件。
# ================================
# ruff: noqa
from experiments.robot.capra.mining.candidate_actions import *  # noqa: F401,F403
