"""Short-horizon and long-horizon counterfactual rollout utilities.

Two rollout modes:

1. short_cf_rollout  (H_s steps)
   Used to evaluate P_t(a) and F_t(a) for a candidate action.
   Called K times per timestep during offline mining.

2. long_replacement_rollout  (W steps, budgeted)
   Used for precursor attribution: replace candidate timestep with
   a safer alternative and measure downstream hazard reduction.
   Called at most `budget` times per dangerous trajectory.

Both modes:
  - restore simulator from Snapshot before executing
  - do NOT modify the original environment state
  - are pure simulation; no model gradient is computed here

Phase 1: interface + stubs.
Phase 2: implement using snapshot.py + state_api.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.env_adapter import CAPRAEnvAdapter
    from experiments.robot.capra.snapshot import Snapshot
    from experiments.robot.capra.state_api import StateSignals
    from experiments.robot.capra.capra_config import CAPRAConfig


@dataclass
class RolloutResult:
    """Outcome of one short counterfactual rollout."""
    action_chunk: np.ndarray          # (chunk_len, action_dim)
    progress: float                   # P_t(a)
    footprint: float                  # F_t(a)
    signals_before: Any               # StateSignals at snap
    signals_after: Any                # StateSignals after H_s steps
    terminated: bool = False
    steps_executed: int = 0


def short_cf_rollout(
    env: "CAPRAEnvAdapter",
    snap: "Snapshot",
    action_chunk: np.ndarray,
    role_map: Any,                    # ObjectRoleMap
    cfg: "CAPRAConfig",
    progress_fn: Optional[Any] = None,
) -> RolloutResult:
    """Branch from snap, execute action_chunk for H_s steps, return metrics.

    Phase 2: restore snapshot, step env, read signals, compute P and F.
    """
    raise NotImplementedError(
        "Phase 2: restore snapshot, execute chunk, call task_progress and footprint."
    )


@dataclass
class LongRolloutResult:
    """Outcome of one long replacement rollout for precursor attribution."""
    replaced_step: int
    replacement_action: np.ndarray    # (chunk_len, action_dim)
    hazard_reduction: float           # downstream hazard delta
    attribution_score: float          # R_t for this step
    counterfactual_trajectory: List[Dict] = field(default_factory=list)


def long_replacement_rollout(
    env: "CAPRAEnvAdapter",
    trajectory_snaps: List["Snapshot"],   # one snap per step in window W
    trajectory_actions: np.ndarray,       # (W, chunk_len, action_dim)
    replacement_step: int,
    replacement_action: np.ndarray,       # (chunk_len, action_dim)
    role_map: Any,
    cfg: "CAPRAConfig",
) -> LongRolloutResult:
    """Replace one action in a window-W trajectory and measure hazard change.

    Phase 2: implement budgeted W-step replacement rollout.
    """
    raise NotImplementedError(
        "Phase 2: implement budgeted precursor attribution rollout."
    )
