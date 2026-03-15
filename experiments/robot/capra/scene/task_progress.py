# ===== CAPRA 任务进度估计 (task_progress.py) =====
#
# 作用
# ----
# 计算 P_t(a)：从状态 s_t 出发执行动作 a（H_s 步）后的任务进度变化量。
# P_t 是等价集 E_t 过滤的核心依据：只有进度足够接近 P_max 的候选才进入 E_t。
#
# 重要设计原则
# ------------
#   P_t 与 F_t（足迹）完全解耦：
#   一个更安全的动作可以和不安全的动作有完全相同的 P_t，
#   这正是 CAPRA 存在的意义——在等价进度下选更安全的动作。
#
# 三种精度的实现
# --------------
#   libero_info_progress     精确：从 env.step 返回的 info 字典读取子任务完成数
#                            需要 info['num_satisfied_predicates'] 字段
#   libero_obs_stage_progress 近似：从物体位置推断任务阶段
#   done_flag_progress       最粗糙：任务成功=1.0，否则=0.5
#                            当前默认使用这个，因为 LIBERO info 不总是暴露子任务数
#
# 主入口
# ------
#   progress_fn = make_libero_progress_fn()
#   result = compute_progress_from_rollout(obs_before, info_before, obs_after, info_after,
#                                          task_description, progress_fn)
#   print(result.value)  # float in [0, 1]

"""Task progress computation: P_t(a).

P_t(a) is the change in progress potential after executing action `a`
from state s_t for H_s steps.

P_t is NOT a raw reward. It uses task-own stage signals or a
decomposable proxy. Coupling to footprint is zero by design: a safer
action may have the same progress as an unsafe one.

Progress API
------------
The primary interface is `ProgressFn`: a callable
  (obs: Dict, task_description: str, info: Dict) -> float in [0, 1].

Built-in implementations (ordered by fidelity):

1. libero_info_progress     EXACT when env exposes subtask counts in info.
   Reads info['num_satisfied_predicates'] / info['num_predicates'].

2. libero_obs_stage_progress  APPROX: infers stage from object position
   changes relative to goal region heuristic.

3. task_keyword_proxy       APPROX: maps task keyword to a rough stage
   estimate based on whether key objects are near their goal positions.
   Fallback when neither info nor goal region is available.

Decoupling from footprint
-------------------------
ProgressResult.value is computed purely from task completion signals.
FootprintComponents are computed separately in footprint.py.
Neither calls the other -- they share only StateSignals as input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.core.signals import StateSignals

# Type alias for a progress function
ProgressFn = Callable[[Dict, str, Dict], float]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProgressResult:
    """Output of a progress computation."""
    value: float           # P_t(a) in [0, 1]
    stage_before: int      # task stage index before rollout  (-1 = unknown)
    stage_after: int       # task stage index after rollout   (-1 = unknown)
    max_stages: int        # total number of stages           (-1 = unknown)
    method: str = ""       # which progress_fn was used
    is_approximate: bool = False

    def delta(self) -> float:
        """Stage change as a fraction of total stages."""
        if self.max_stages <= 0:
            return self.value
        return (self.stage_after - self.stage_before) / self.max_stages


# ---------------------------------------------------------------------------
# 1. Primary: LIBERO info-dict progress (EXACT)
# ---------------------------------------------------------------------------

def libero_info_progress(obs: Dict, task_description: str, info: Dict) -> float:
    """Read task progress from LIBERO env info dict.

    LIBERO returns an `info` dict from env.step() that includes:
      info['num_satisfied_predicates']  -- int, current satisfied sub-goals
      info['num_predicates']            -- int, total sub-goals

    Returns satisfied / total in [0, 1].
    Returns -1.0 if the required keys are absent (caller should fall back).

    Fidelity: EXACT (uses simulator's own predicate checker).
    """
    n_sat = info.get("num_satisfied_predicates", None)
    n_total = info.get("num_predicates", None)
    if n_sat is None or n_total is None or n_total == 0:
        return -1.0
    return float(n_sat) / float(n_total)


# ---------------------------------------------------------------------------
# 2. Secondary: object-displacement proxy (APPROX)
# ---------------------------------------------------------------------------

def object_proximity_progress(
    poses_current: Dict[str, Any],   # Dict[str, ObjectPose]
    goal_positions: Dict[str, np.ndarray],
    tolerance_m: float = 0.05,
) -> float:
    """Estimate progress as fraction of goal objects within tolerance.

    Args:
        poses_current: Current object poses from read_object_poses().
        goal_positions: {object_name: goal_position_array (3,)} dict.
        tolerance_m: Distance threshold to count object as 'at goal'.

    Returns fraction in [0, 1]; 0.0 if goal_positions is empty.

    Fidelity: APPROXIMATION (requires pre-specified goal positions).
    """
    if not goal_positions:
        return 0.0
    n_satisfied = 0
    for name, goal_pos in goal_positions.items():
        if name not in poses_current:
            continue
        dist = float(np.linalg.norm(
            poses_current[name].position - np.asarray(goal_pos)
        ))
        if dist <= tolerance_m:
            n_satisfied += 1
    return n_satisfied / len(goal_positions)


# ---------------------------------------------------------------------------
# 3. Fallback: z-height proxy for pick tasks (APPROX)
# ---------------------------------------------------------------------------

def pick_height_proxy(
    obs: Dict,
    target_name: str,
    lift_height_m: float = 0.10,
    table_z: float = 0.82,
) -> float:
    """Estimate pick-task progress from target object height above table.

    0.0  = object at table level
    0.5  = object at 50% of lift_height
    1.0  = object at or above lift_height

    Fidelity: APPROXIMATION (height proxy, not goal check).
    Works only for pick/lift tasks; returns 0.0 for other task types.
    """
    pos_key = target_name + "_pos"
    if pos_key not in obs:
        return 0.0
    pos = np.asarray(obs[pos_key]).flatten()
    if pos.shape != (3,):
        return 0.0
    height_above_table = max(0.0, pos[2] - table_z)
    return float(np.clip(height_above_table / lift_height_m, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Composite progress function builder
# ---------------------------------------------------------------------------

def make_libero_progress_fn(
    goal_positions: Optional[Dict[str, np.ndarray]] = None,
    target_name: Optional[str] = None,
) -> ProgressFn:
    """Build a ProgressFn for LIBERO tasks.

    Priority order:
      1. libero_info_progress (uses env info dict -- EXACT)
      2. object_proximity_progress (uses goal_positions -- APPROX)
      3. pick_height_proxy (uses target z-height -- APPROX)
      4. Returns 0.0 as last resort

    The caller can pass goal_positions and/or target_name to enable
    fallback levels 2 and 3 respectively.
    """
    def _fn(obs: Dict, task_description: str, info: Dict) -> float:
        # Level 1: exact info-dict signal
        v = libero_info_progress(obs, task_description, info)
        if v >= 0.0:
            return v

        # Level 2: proximity to goal positions
        if goal_positions:
            from experiments.robot.capra.core.signals import read_object_poses
            poses = read_object_poses(obs)
            return object_proximity_progress(poses, goal_positions)

        # Level 3: height proxy for pick tasks
        if target_name and any(kw in task_description.lower()
                               for kw in ("pick", "lift", "grasp", "grab")):
            return pick_height_proxy(obs, target_name)

        return 0.0

    return _fn


# ---------------------------------------------------------------------------
# ProgressResult builder from a rollout
# ---------------------------------------------------------------------------

def compute_progress_from_rollout(
    obs_before: Dict,
    info_before: Dict,
    obs_after: Dict,
    info_after: Dict,
    task_description: str,
    progress_fn: Optional[ProgressFn] = None,
) -> ProgressResult:
    """Compute P_t(a) = progress_after - progress_before.

    Both observations come from a short CF rollout (H_s steps).
    The progress_fn is evaluated on both endpoints; the difference
    is the progress contribution of action `a`.

    If progress_fn is None, uses make_libero_progress_fn() with
    no goal positions (falls through to info-dict only).

    Returns ProgressResult with value = after - before, clipped to [0, 1].
    Negative delta is clipped to 0 (regression is not rewarded).
    """
    if progress_fn is None:
        progress_fn = make_libero_progress_fn()

    # Determine stage counts from info if available
    n_total = info_after.get("num_predicates", -1)
    n_before = info_before.get("num_satisfied_predicates", -1)
    n_after  = info_after.get("num_satisfied_predicates", -1)

    p_before = progress_fn(obs_before, task_description, info_before)
    p_after  = progress_fn(obs_after,  task_description, info_after)

    # Use info-based stages if exact values are available
    if n_before >= 0 and n_after >= 0 and n_total > 0:
        stage_before = n_before
        stage_after  = n_after
        is_approx    = False
        method       = "libero_info"
    else:
        stage_before = -1
        stage_after  = -1
        is_approx    = True
        method       = "proxy"

    value = float(np.clip(p_after - p_before, 0.0, 1.0))

    return ProgressResult(
        value=value,
        stage_before=stage_before,
        stage_after=stage_after,
        max_stages=n_total,
        method=method,
        is_approximate=is_approx,
    )
