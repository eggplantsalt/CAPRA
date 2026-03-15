# ===== 向后兼容导入 shim =====
# 本文件是一个转发层（shim），实际代码已迁移到子包中。
# 保留本文件是为了让旧的导入语句继续工作，例如：
#   from experiments.robot.capra.capra_config import ...  ← 仍然有效
#
# 功能说明：CAPRA 超参数配置中心，所有算法参数的唯一入口
#
# 实际实现位置：experiments.robot.capra.core.capra_config
# 如需查看或修改代码，请打开上方的实际实现文件。
# ================================
# ruff: noqa
from experiments.robot.capra.core.capra_config import *  # noqa: F401,F403
