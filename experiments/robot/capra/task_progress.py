"""Task progress computation: P_t(a).

P_t(a) is the change in progress potential after executing action `a`
from state `s_t` for H_s steps in the simulator.

Progress is NOT a raw reward; it uses the task's own stage/phase
signals or a decomposable progress potential function.

Phase 1: interface + stub.
Phase 2: implement LIBERO stage-signal reader and potential function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.env_adapter import CAPRAEnvAdapter
    from experiments.robot.capra.snapshot import Snapshot


@dataclass
class ProgressResult:
    value: float          # P_t(a) in [0, 1]
    stage_before: int     # task stage index before rollout
    stage_after: int      # task stage index after rollout
    is_approximate: bool = False


def compute_progress(
    env: "CAPRAEnvAdapter",
    snap: "Snapshot",
    action_chunk: np.ndarray,
    H_s: int,
    task_description: str,
    progress_fn: Optional[Any] = None,
) -> ProgressResult:
    """Execute action_chunk from snap for H_s steps and return progress.

    Args:
        env: CAPRA env adapter (will be restored to snap before rollout).
        snap: Saved simulator state to branch from.
        action_chunk: Action chunk shape (chunk_len, action_dim).
        H_s: Short counterfactual horizon in environment steps.
        task_description: Natural language task string.
        progress_fn: Optional callable(obs, task_description) -> float.
                     If None, falls back to LIBERO task-stage signal.

    Phase 2: implement restore + step loop + stage reader.
    """
    raise NotImplementedError("Phase 2: implement short-horizon rollout + stage reader.")


def libero_stage_progress(obs: Dict, task_description: str) -> float:
    """Map LIBERO task-stage completion signal to a scalar in [0, 1].

    Phase 2: read from env info dict or BDDL predicate checker.
    """
    raise NotImplementedError("Phase 2.")
