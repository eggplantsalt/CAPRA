# ===== 向后兼容导入 shim =====
# 本文件是一个转发层（shim），实际代码已迁移到子包中。
# 保留本文件是为了让旧的导入语句继续工作，例如：
#   from experiments.robot.capra.run_capra_eval import ...  ← 仍然有效
#
# 功能说明：CAPRA 评估主循环入口，收集 SPIR/EAR 等安全指标
#
# 实际实现位置：experiments.robot.capra.eval.run_capra_eval
# 如需查看或修改代码，请打开上方的实际实现文件。
# ================================
# ruff: noqa
from experiments.robot.capra.eval.run_capra_eval import *  # noqa: F401,F403
