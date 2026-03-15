# ===== CAPRA 状态信号 API 门面 (state_api.py) =====
#
# 作用
# ----
# 向后兼容的公共导出层。
# 所有信号数据类和读取函数的实际实现在 signals.py 中。
# 本文件只做 re-export，确保旧的导入路径仍然有效：
#   from experiments.robot.capra.core.state_api import StateSignals  # 仍然可用
#
# 如果你要修改信号逻辑，请修改 signals.py，不要改本文件。

"""Environment state signal API -- public facade.

All signal dataclasses and reader functions are implemented in
`signals.py`.  This module re-exports them under the original names
so existing imports of `state_api.ObjectPose` etc. keep working.

Added here: `read_state_signals` now delegates to `signals.read_state_signals`
with the full implementation instead of raising NotImplementedError.
"""
from __future__ import annotations

# Re-export all public names from signals so callers can do:
#   from experiments.robot.capra.core.state_api import StateSignals, ObjectPose ...
from experiments.robot.capra.core.signals import (  # noqa: F401
    ObjectPose,
    ContactEvent,
    SupportRelation,
    StateSignals,
    DEFAULT_WORKSPACE_BOUNDS,
    read_object_poses,
    read_contacts,
    read_support_relations,
    read_topple_flags,
    check_workspace_violations,
    read_state_signals,
)
