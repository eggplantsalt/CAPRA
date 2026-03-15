"""Task-equivalent action set E_t.

An action is in E_t iff:
  1. P_t(a) >= progress_floor               (minimum progress gate)
  2. |P_max - P_t(a)| <= epsilon_p_abs      (absolute gap)
  3. |P_max - P_t(a)| / P_max <= epsilon_p_rel  (relative gap)

Only actions passing all three criteria are compared for safety.
If E_t is empty, the CAPRA loss is NOT triggered for this timestep.

Phase 1: filter logic (pure Python / numpy, no env dependency).
"""
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.capra_config import CAPRAConfig


def build_task_equivalent_set(
    candidate_actions: np.ndarray,   # (K, chunk_len, action_dim)
    progress_values: np.ndarray,     # (K,)  P_t(a_i) for each candidate
    cfg: "CAPRAConfig",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (equivalent_actions, equivalent_indices, P_max).

    Returns empty arrays when the set is empty (CAPRA loss not triggered).
    """
    K = len(progress_values)
    assert candidate_actions.shape[0] == K, "actions and progress must have same length"

    P_max = float(np.max(progress_values))

    # Gate 1: minimum progress
    if P_max < cfg.progress_floor:
        empty = np.empty((0,) + candidate_actions.shape[1:], dtype=candidate_actions.dtype)
        return empty, np.array([], dtype=int), P_max

    # Gates 2 & 3: task-equivalence
    abs_gap = np.abs(P_max - progress_values)
    rel_gap = abs_gap / (P_max + 1e-8)

    mask = (
        (progress_values >= cfg.progress_floor)
        & (abs_gap <= cfg.epsilon_p_abs)
        & (rel_gap <= cfg.epsilon_p_rel)
    )

    indices = np.where(mask)[0]
    return candidate_actions[indices], indices, P_max


def local_safest_action_index(
    equivalent_indices: np.ndarray,  # indices into original candidate array
    footprint_values: np.ndarray,    # (K,)  F_t(a_i) for all candidates
) -> int:
    """Return the index (in the original K-candidate array) with minimum footprint
    among the task-equivalent set.
    """
    if len(equivalent_indices) == 0:
        raise ValueError("equivalent_indices is empty – CAPRA loss should not be triggered.")
    best_local = int(np.argmin(footprint_values[equivalent_indices]))
    return int(equivalent_indices[best_local])


def compute_local_avoidable_risk(
    chosen_footprint: float,
    min_equivalent_footprint: float,
) -> float:
    """Delta_t = F_t(chosen) - F_t(safest_equivalent).

    This is the avoidable excess risk at timestep t.
    Clipped at 0 so that Delta_t is non-negative.
    """
    return max(0.0, chosen_footprint - min_equivalent_footprint)
