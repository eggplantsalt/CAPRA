"""Short-horizon counterfactual rollout for CAPRA.

For each candidate action chunk:
  1. Restore the env to a saved Snapshot.
  2. Execute the chunk for min(chunk_len, H_s) steps.
  3. Read StateSignals before and after.
  4. Compute P_t(a) and F_t(a) with decompositions.
  5. Return a RolloutResult.

Also provides TimestepRecord -- the unit of the offline mining cache.

Design constraints
------------------
- Never modifies the original env state; always restores before each rollout.
- No model gradients computed here.
- long_replacement_rollout stubbed for Phase 4 (precursor attribution).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.mining.env_adapter import CAPRAEnvAdapter
    from experiments.robot.capra.mining.snapshot import Snapshot
    from experiments.robot.capra.core.signals import StateSignals
    from experiments.robot.capra.scene.object_roles import ObjectRoleMap
    from experiments.robot.capra.core.capra_config import CAPRAConfig
    from experiments.robot.capra.scene.task_progress import ProgressFn


# ---------------------------------------------------------------------------
# Single rollout result
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """Outcome of one short counterfactual rollout."""
    action_chunk: np.ndarray       # (chunk_len, action_dim)
    progress: float                # P_t(a) in [0, 1]
    footprint: float               # F_t(a) scalar
    footprint_components: Any      # FootprintComponents
    signals_before: Any            # StateSignals at snap
    signals_after: Any             # StateSignals after rollout
    obs_after: Dict
    info_after: Dict
    terminated: bool = False
    steps_executed: int = 0


# ---------------------------------------------------------------------------
# Short CF rollout
# ---------------------------------------------------------------------------

def short_cf_rollout(
    env: "CAPRAEnvAdapter",
    snap: "Snapshot",
    action_chunk: np.ndarray,
    role_map: "ObjectRoleMap",
    cfg: "CAPRAConfig",
    progress_fn: Optional["ProgressFn"] = None,
    object_names: Optional[List[str]] = None,
) -> RolloutResult:
    """Branch from snap, execute action_chunk for H_s steps, return metrics."""
    from experiments.robot.capra.mining.snapshot import restore_snapshot
    from experiments.robot.capra.core.signals import read_state_signals
    from experiments.robot.capra.core.footprint import aggregate_footprint_components, compute_footprint
    from experiments.robot.capra.scene.task_progress import make_libero_progress_fn, compute_progress_from_rollout

    if progress_fn is None:
        progress_fn = make_libero_progress_fn()

    n_steps = min(action_chunk.shape[0], cfg.H_s)

    # 1. Restore snapshot
    restore_snapshot(env, snap)

    # 2. Signals before
    signals_before = read_state_signals(
        snap.obs, step=snap.step, env=env, object_names=object_names
    )

    # 3. Execute
    obs, info = snap.obs, snap.info
    terminated = False
    steps_executed = 0
    for i in range(n_steps):
        obs, _reward, done, info = env.step(action_chunk[i])
        steps_executed += 1
        if done:
            terminated = True
            break

    # 4. Signals after
    signals_after = read_state_signals(
        obs, step=snap.step + steps_executed, env=env,
        object_names=object_names, poses_before=signals_before.object_poses
    )

    # 5. Progress
    progress_result = compute_progress_from_rollout(
        snap.obs, snap.info, obs, info, snap.task_description, progress_fn
    )

    # 6. Footprint
    components = aggregate_footprint_components(signals_before, signals_after, role_map)
    fp_scalar  = compute_footprint(components, cfg)

    return RolloutResult(
        action_chunk=action_chunk,
        progress=progress_result.value,
        footprint=fp_scalar,
        footprint_components=components,
        signals_before=signals_before,
        signals_after=signals_after,
        obs_after=obs,
        info_after=info,
        terminated=terminated,
        steps_executed=steps_executed,
    )


# ---------------------------------------------------------------------------
# TimestepRecord: intermediate mining cache entry
# ---------------------------------------------------------------------------

@dataclass
class TimestepRecord:
    """All intermediate quantities for one analysed timestep.

    Stored in the offline mining cache; contains everything needed to
    compute CAPRA loss without re-running the simulator.
    """
    # Identity
    episode_id: str
    step: int

    # Candidates
    candidate_actions: np.ndarray   # (K, chunk_len, action_dim)
    prior_weights: np.ndarray       # (K,)  uniform unless replaced

    # Per-candidate rollout metrics
    progress_values: np.ndarray     # (K,)  P_t(a_i)
    footprint_values: np.ndarray    # (K,)  F_t(a_i)
    footprint_components: List[Any] # List[FootprintComponents] length K

    # Equivalence set
    equivalent_indices: np.ndarray  # indices of E_t candidates
    p_max: float

    # Risk quantities
    delta_t: float                  # local avoidable risk
    chosen_action_idx: int = 0      # index of policy's nominal action (always 0)

    # Optional embedding for buffer retrieval
    obs_embedding: Optional[np.ndarray] = None   # (D,)

    # Metadata
    task_description: str = ""


# ---------------------------------------------------------------------------
# Mine one timestep: run K rollouts + build TimestepRecord
# ---------------------------------------------------------------------------

def mine_one_timestep(
    env: "CAPRAEnvAdapter",
    snap: "Snapshot",
    candidate_actions: np.ndarray,   # (K, chunk_len, action_dim)
    prior_weights: np.ndarray,       # (K,)
    role_map: "ObjectRoleMap",
    cfg: "CAPRAConfig",
    episode_id: str = "unknown",
    progress_fn: Optional["ProgressFn"] = None,
    object_names: Optional[List[str]] = None,
) -> TimestepRecord:
    """Run K short CF rollouts and build a TimestepRecord.

    This is the core of the offline mining loop.  Called once per
    timestep per episode during mining.

    Returns a fully populated TimestepRecord.
    """
    from experiments.robot.capra.core.equivalence import (
        build_task_equivalent_set,
        local_safest_action_index,
        compute_local_avoidable_risk,
    )

    K = len(candidate_actions)
    progress_values  = np.zeros(K, dtype=np.float32)
    footprint_values = np.zeros(K, dtype=np.float32)
    footprint_components: List[Any] = []

    for i in range(K):
        result = short_cf_rollout(
            env, snap, candidate_actions[i], role_map, cfg,
            progress_fn=progress_fn, object_names=object_names
        )
        progress_values[i]  = result.progress
        footprint_values[i] = result.footprint
        footprint_components.append(result.footprint_components)

    # Build equivalence set
    _eq_actions, eq_idx, p_max = build_task_equivalent_set(
        candidate_actions, progress_values, cfg
    )

    # Local avoidable risk
    if len(eq_idx) > 0:
        safest_idx = local_safest_action_index(eq_idx, footprint_values)
        delta_t = compute_local_avoidable_risk(
            chosen_footprint=float(footprint_values[0]),  # index 0 = nominal
            min_equivalent_footprint=float(footprint_values[safest_idx]),
        )
    else:
        delta_t = 0.0

    return TimestepRecord(
        episode_id=episode_id,
        step=snap.step,
        candidate_actions=candidate_actions,
        prior_weights=prior_weights,
        progress_values=progress_values,
        footprint_values=footprint_values,
        footprint_components=footprint_components,
        equivalent_indices=eq_idx,
        p_max=p_max,
        delta_t=delta_t,
        task_description=snap.task_description,
    )


# ---------------------------------------------------------------------------
# Long replacement rollout (Phase 4 stub)
# ---------------------------------------------------------------------------

@dataclass
class LongRolloutResult:
    replaced_step: int
    replacement_action: np.ndarray
    hazard_reduction: float
    attribution_score: float
    counterfactual_trajectory: List[Dict] = field(default_factory=list)


def long_replacement_rollout(
    env: "CAPRAEnvAdapter",
    trajectory_snaps: List["Snapshot"],
    trajectory_actions: np.ndarray,
    replacement_step: int,
    replacement_action: np.ndarray,
    role_map: Any,
    cfg: "CAPRAConfig",
) -> LongRolloutResult:
    """Phase 4 stub: precursor attribution long rollout."""
    raise NotImplementedError(
        "Phase 4: implement budgeted precursor attribution rollout."
    )
