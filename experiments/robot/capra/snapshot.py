# ===== 向后兼容导入 shim =====
# 本文件是一个转发层（shim），实际代码已迁移到子包中。
# 保留本文件是为了让旧的导入语句继续工作，例如：
#   from experiments.robot.capra.snapshot import ...  ← 仍然有效
#
# 功能说明：MuJoCo 状态快照/还原，让 K 个候选从同一起点出发
#
# 实际实现位置：experiments.robot.capra.mining.snapshot
# 如需查看或修改代码，请打开上方的实际实现文件。
# ================================
# ruff: noqa
from experiments.robot.capra.mining.snapshot import *  # noqa: F401,F403
