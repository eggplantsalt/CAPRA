"""Snapshot / restore for CAPRA short-horizon counterfactual rollouts.

Saves and restores the full MuJoCo simulator state so that K candidate
actions can each branch from the exact same starting point.

Backend
-------
LIBERO wraps robosuite which wraps MjSim (mujoco-py) or MjModel/MjData
(mujoco >= 3).  Both expose a round-trip state object:

  mujoco-py:  sim.get_state()  -> MjSimState  (qpos, qvel, act, udd_state)
              sim.set_state(state) + sim.forward()

  mujoco 3:   mujoco.MjData  -- copy via mujoco.mj_copyData

We try the mujoco-py path first (LIBERO default), then fall back to a
obs-dict snapshot that re-initialises the scene from a recorded obs dict
(APPROX -- loses velocity state, fine for H_s <= 5).

Fidelity levels
---------------
EXACT   full qpos + qvel + act captured via sim.get_state()
APPROX  position-only re-init from obs dict (loses velocity)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.mining.env_adapter import CAPRAEnvAdapter


class SnapshotFidelity:
    EXACT  = "exact"   # full MuJoCo qpos/qvel + act
    APPROX = "approx"  # obs-dict position re-init


@dataclass
class Snapshot:
    """Opaque container for a saved simulator state."""
    fidelity: str
    data: Any           # MjSimState | dict
    step: int
    obs: Dict           # last obs dict (always stored for fallback + signal reading)
    info: Dict          # last info dict (for progress computation)
    task_description: str = ""


def save_snapshot(
    env: "CAPRAEnvAdapter",
    obs: Dict,
    info: Dict,
    step: int,
    task_description: str = "",
) -> Snapshot:
    """Capture current simulator state.

    Tries EXACT MuJoCo state first; falls back to obs-dict APPROX.
    """
    try:
        sim = env._env.env.sim     # OffScreenRenderEnv -> robosuite -> MjSim
        state = sim.get_state()
        return Snapshot(
            fidelity=SnapshotFidelity.EXACT,
            data=state,
            step=step,
            obs=obs,
            info=info,
            task_description=task_description,
        )
    except AttributeError:
        # Sim not accessible (unit tests, or non-MuJoCo backend)
        return Snapshot(
            fidelity=SnapshotFidelity.APPROX,
            data={"obs": obs},
            step=step,
            obs=obs,
            info=info,
            task_description=task_description,
        )


def restore_snapshot(env: "CAPRAEnvAdapter", snap: Snapshot) -> None:
    """Restore simulator to a previously saved snapshot.

    After this call the env is ready for a fresh rollout from snap.step.
    """
    if snap.fidelity == SnapshotFidelity.EXACT:
        try:
            sim = env._env.env.sim
            sim.set_state(snap.data)
            sim.forward()   # propagate kinematics
            return
        except AttributeError:
            pass  # fall through to APPROX

    # APPROX fallback: cannot truly restore; caller must handle gracefully
    # (short rollouts with H_s <= 5 are still useful even with v=0 init)
    raise RuntimeError(
        "APPROX snapshot restore not supported for live env. "
        "Ensure the env exposes env.env.sim (mujoco-py MjSim API)."
    )
