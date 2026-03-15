"""Simulator snapshot / restore interface.

Provides save_snapshot() / restore_snapshot() that allow CAPRA's
short-horizon counterfactual rollout to branch from a fixed state.

Phase 1: interface + fidelity constants only.
Phase 2: fill MuJoCo / LIBERO backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.robot.capra.env_adapter import CAPRAEnvAdapter


# Fidelity levels – recorded here so approximation choices are explicit.
class SnapshotFidelity:
    EXACT = "exact"          # full MuJoCo qpos/qvel + object poses
    APPROX = "approx"        # re-create scene from deterministic initial state


@dataclass
class Snapshot:
    """Opaque container for a saved simulator state."""
    fidelity: str
    data: Any           # backend-specific (MjSimState or dict)
    step: int           # environment step index at capture time
    task_description: str = ""


def save_snapshot(env: "CAPRAEnvAdapter", step: int = 0) -> Snapshot:
    """Capture current simulator state.

    Tries EXACT first; falls back to APPROX if the backend does not
    expose a full state round-trip.
    """
    raise NotImplementedError(
        "Phase 2: call env.get_sim_state() and pack into Snapshot."
    )


def restore_snapshot(env: "CAPRAEnvAdapter", snap: Snapshot) -> None:
    """Restore simulator to a previously saved snapshot.

    After this call the environment is ready for a fresh rollout
    starting from `snap.step`.
    """
    raise NotImplementedError(
        "Phase 2: call env.set_sim_state(snap.data)."
    )
