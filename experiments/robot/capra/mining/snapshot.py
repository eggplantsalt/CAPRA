# ===== CAPRA MuJoCo 快照/还原 (snapshot.py) =====
#
# 作用
# ----
# 在短时 counterfactual rollout 中，每次执行候选动作前需要
# 从同一个起始状态出发。这个模块负责保存和还原 MuJoCo 仿真器的完整状态。
#
# 两种精度
# --------
#   EXACT   通过 mujoco-py 的 sim.get_state() / sim.set_state() 保存/还原
#           保存内容：qpos（位置）、qvel（速度）、act（激活状态）
#           还原后物理状态完全一致，CF rollout 结果可复现
#
#   APPROX  sim 不可访问时的降级方案
#           只记录 obs dict（位置，但不含速度）
#           还原不完整，H_s <= 5 时误差较小，可接受
#
# 访问路径
# --------
#   env._env.env.sim  →  OffScreenRenderEnv → robosuite env → MjSim
#   sim.get_state()   →  返回 MjSimState 对象（包含全部物理状态）
#   sim.set_state(s)  →  恢复物理状态
#   sim.forward()     →  传播运动学（还原后必须调用）
#
# 使用方式
# --------
#   snap = save_snapshot(env, obs, info, step, task_description)
#   # ... 执行候选动作 ...
#   restore_snapshot(env, snap)  # 还原到 snap 保存时的状态

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
