# ===== 向后兼容导入 shim =====
# 本文件是一个转发层（shim），实际代码已迁移到子包中。
# 保留本文件是为了让旧的导入语句继续工作，例如：
#   from experiments.robot.capra.report_utils import ...  ← 仍然有效
#
# 功能说明：结果报告生成器，输出 JSON/CSV/Markdown 四种格式
#
# 实际实现位置：experiments.robot.capra.eval.report_utils
# 如需查看或修改代码，请打开上方的实际实现文件。
# ================================
# ruff: noqa
from experiments.robot.capra.eval.report_utils import *  # noqa: F401,F403
