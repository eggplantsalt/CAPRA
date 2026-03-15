"""Environment state signal API -- public facade.

All signal dataclasses and reader functions are implemented in
`signals.py`.  This module re-exports them under the original names
so existing imports of `state_api.ObjectPose` etc. keep working.

Added here: `read_state_signals` now delegates to `signals.read_state_signals`
with the full implementation instead of raising NotImplementedError.
"""
from __future__ import annotations

# Re-export all public names from signals so callers can do:
#   from experiments.robot.capra.state_api import StateSignals, ObjectPose ...
from experiments.robot.capra.signals import (  # noqa: F401
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
