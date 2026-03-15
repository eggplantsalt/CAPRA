"""Precursor attribution: R_t and precursor chain computation.

For a dangerous trajectory, look back W steps and identify which
early action choices were responsible for the downstream hazard.

Approach:
  For each step t' in [t-W, t):
    Replace a_t' with its safest task-equivalent alternative.
    Run a budgeted W-step rollout.
    Measure reduction in downstream hazard: delta_hazard(t').
  Normalise to get attribution scores R_t'.

Outputs:
  - precursor_chain: ordered list of (step, score) pairs
  - R_t: attribution weight for each step (used in training loss weight w_t)

Phase 1: dataclass definitions + stubs.
Phase 2: implement using long_replacement_rollout from rollout.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.env_adapter import CAPRAEnvAdapter
    from experiments.robot.capra.snapshot import Snapshot
    from experiments.robot.capra.capra_config import CAPRAConfig


@dataclass
class PrecursorEntry:
    """Attribution result for one candidate precursor step."""
    step: int
    delta_hazard: float           # how much downstream hazard dropped when this step was replaced
    attribution_score: float      # R_t normalised in [0, 1]
    replacement_action: np.ndarray   # (chunk_len, action_dim) -- safest equivalent used


@dataclass
class PrecursorChain:
    """Full attribution result for one dangerous trajectory segment."""
    anchor_step: int              # the terminal hazard step
    window: int                   # W -- lookback window used
    entries: List[PrecursorEntry] = field(default_factory=list)

    def get_weight(self, step: int) -> float:
        """Return R_t for a given step index; 0.0 if step not in chain."""
        for e in self.entries:
            if e.step == step:
                return e.attribution_score
        return 0.0

    def top_k(self, k: int = 3) -> List[PrecursorEntry]:
        """Return the k entries with highest attribution score."""
        return sorted(self.entries, key=lambda e: e.attribution_score, reverse=True)[:k]


def compute_precursor_chain(
    env: "CAPRAEnvAdapter",
    trajectory_snaps: List["Snapshot"],     # one per step in window W
    trajectory_actions: np.ndarray,          # (W, chunk_len, action_dim)
    trajectory_footprints: np.ndarray,       # (W,)  F_t for each step
    anchor_step: int,
    role_map: Any,                           # ObjectRoleMap
    cfg: "CAPRAConfig",
    budget: int = 5,                         # max replacement rollouts
) -> PrecursorChain:
    """Compute precursor attribution chain for a dangerous trajectory.

    For each step t' in the window, estimate how much downstream hazard
    would have been avoided by substituting a safer task-equivalent action.
    Returns a PrecursorChain with normalised attribution scores R_t'.

    Phase 2: implement using rollout.long_replacement_rollout.
    """
    raise NotImplementedError(
        "Phase 2: implement budgeted precursor attribution using "
        "rollout.long_replacement_rollout."
    )


def precursor_loss_weight(delta_t: float, r_t: float, rho: float) -> float:
    """Compute per-timestep training loss weight.

    w_t = Delta_t * (1 + rho * R_t)

    Args:
        delta_t: Local avoidable risk at step t.
        r_t: Precursor attribution score R_t from PrecursorChain.
        rho: Upweight factor from CAPRAConfig.
    """
    return delta_t * (1.0 + rho * r_t)
