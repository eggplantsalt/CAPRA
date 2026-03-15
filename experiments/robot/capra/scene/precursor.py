# ===== CAPRA 前驱归因 (precursor.py) =====
#
# 核心思想
# --------
# 很多危险事件不是"最后一步"造成的，而是更早的决策埋下了祸根。
# 前驱归因找到"替换哪一步最能减少最终危险"，并对那些步骤的损失权重上调。
#
# 算法流程
# --------
#   输入：危险轨迹窗口 W（以高危步骤 anchor_step 结尾）
#   对窗口内每步 t'（按原始足迹从高到低排序，预算优先高危步骤）：
#     - 在 t' 处替换为等价集中最安全的动作
#     - 执行 H_attr 步后计算下游总足迹
#     - delta_hazard(t') = max(0, hazard_before - hazard_after)
#   R_t' = delta_hazard(t') / sum(all deltas)  [归一化到 0-1]
#
# 损失权重加成
# ------------
#   w_t = Delta_t * (1 + rho * R_t)
#   rho > 0 让前驱步骤的损失权重高于非前驱步骤
#
# 两种实现
# --------
#   compute_precursor_chain_from_footprints()  纯 numpy，用于测试和离线分析
#   compute_precursor_chain()                  基于真实 env rollout（需要 sim.get_state）
#   precursor_loss_weight()                    计算 w_t = Delta_t * (1 + rho * R_t)

"""Precursor attribution: R_t, PrecursorChain, AttributionEditGain.

Algorithm (pseudo-code)
-----------------------
Input: dangerous trajectory window W ending at anchor step T
       (F_T >= hazard_threshold)

For t' in window (up to attribution_max_steps, highest-F first):
    candidates = safest task-equivalent replacements from TimestepRecord
    For each r in candidates[:attribution_max_replacements]:
        restore snapshot at t'
        execute r for attribution_rollout_len steps
        replay original actions for remaining steps
        h_after = sum(F) over executed steps
    delta_hazard(t') = max(0, h_before(t') - min(h_after))

R_t' = delta_hazard(t') / (sum_all_deltas + eps)  [normalised]

Budget controls (all in CAPRAConfig)
--------------------------------------
  W                             lookback window (default 10)
  attribution_max_steps         max steps analysed per trajectory (default 10)
  attribution_max_replacements  max candidates tried per step (default 4)
  attribution_rollout_len       steps per replacement rollout (default 8)
  attribution_hazard_threshold  min F_t to trigger (default 0.10)

How we avoid computation explosion
------------------------------------
1. attribution_max_steps caps steps scanned.
2. attribution_max_replacements caps inner loop per step.
3. attribution_rollout_len caps rollout length.
4. Steps are sorted by descending original F_t; budget is spent on
   most dangerous steps first.
5. Attribution is an offline mining pass -- not run at training time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.mining.env_adapter import CAPRAEnvAdapter
    from experiments.robot.capra.mining.snapshot import Snapshot
    from experiments.robot.capra.core.capra_config import CAPRAConfig
    from experiments.robot.capra.scene.object_roles import ObjectRoleMap


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PrecursorEntry:
    """Attribution result for one candidate precursor step."""
    step: int
    delta_hazard: float           # hazard reduction when this step was replaced
    attribution_score: float      # R_t' normalised in [0, 1]
    replacement_action: np.ndarray   # (chunk_len, action_dim)


@dataclass
class PrecursorChain:
    """Full attribution result for one dangerous trajectory segment."""
    anchor_step: int
    window: int
    entries: List[PrecursorEntry] = field(default_factory=list)

    def get_weight(self, step: int) -> float:
        """R_t' for a given step; 0.0 if not in chain."""
        for e in self.entries:
            if e.step == step:
                return e.attribution_score
        return 0.0

    def top_k(self, k: int = 3) -> List[PrecursorEntry]:
        return sorted(self.entries, key=lambda e: -e.attribution_score)[:k]

    def is_empty(self) -> bool:
        return len(self.entries) == 0


# ---------------------------------------------------------------------------
# Loss weight
# ---------------------------------------------------------------------------

def precursor_loss_weight(delta_t: float, r_t: float, rho: float) -> float:
    """w_t = Delta_t * (1 + rho * R_t)"""
    return float(delta_t * (1.0 + rho * r_t))


# ---------------------------------------------------------------------------
# Pure-numpy downstream hazard helper
# ---------------------------------------------------------------------------

def measure_downstream_hazard(
    footprint_sequence: np.ndarray,  # (T,)
    start_idx: int,
) -> float:
    """Sum of footprints from start_idx to end of sequence."""
    if start_idx >= len(footprint_sequence):
        return 0.0
    return float(footprint_sequence[start_idx:].sum())


# ---------------------------------------------------------------------------
# Synthetic attribution (no live env -- for testing and offline analysis)
# ---------------------------------------------------------------------------

def compute_precursor_chain_from_footprints(
    step_footprints: np.ndarray,       # (W,) F_t for each step in window
    replacement_footprints: np.ndarray, # (W,) hypothetical F_t after replacement
    anchor_step: int,
    window_start_step: int,
    cfg: Any,  # CAPRAConfig
) -> "PrecursorChain":
    """Compute PrecursorChain from pre-computed footprint arrays.

    This is the self-contained, env-free version used in tests and
    for post-hoc analysis of cached mining data.

    Args:
        step_footprints:       original F_t for each step in the window (W,)
        replacement_footprints: F_t after the best replacement at each step (W,)
        anchor_step:           absolute step index of the hazard
        window_start_step:     absolute step index of window start
        cfg:                   CAPRAConfig (uses attribution_max_steps)

    Hazard reduction at step i:
        suffix_before[i] = sum(step_footprints[i:])
        suffix_after[i]  = replacement_footprints[i] + sum(step_footprints[i+1:])
        delta[i]         = max(0, suffix_before[i] - suffix_after[i])
                         = max(0, step_footprints[i] - replacement_footprints[i])
    """
    W = len(step_footprints)
    max_steps = min(W, getattr(cfg, 'attribution_max_steps', W))

    # Prioritise by descending original footprint
    step_order = np.argsort(-step_footprints)[:max_steps]

    raw: List[Tuple[int, float]] = []
    for idx in step_order:
        orig = float(step_footprints[idx])
        repl = float(replacement_footprints[idx])
        delta = max(0.0, orig - repl)
        if delta > 0:
            raw.append((int(idx), delta))

    if not raw:
        return PrecursorChain(anchor_step=anchor_step, window=W)

    total = sum(d for _, d in raw) + 1e-12
    entries = [
        PrecursorEntry(
            step=window_start_step + idx,
            delta_hazard=delta,
            attribution_score=delta / total,
            replacement_action=np.zeros((1, 7), dtype=np.float32),
        )
        for idx, delta in sorted(raw, key=lambda x: x[0])
    ]
    return PrecursorChain(anchor_step=anchor_step, window=W, entries=entries)


# ---------------------------------------------------------------------------
# Env-based replacement rollout
# ---------------------------------------------------------------------------

def _replacement_downstream_hazard(
    env: "CAPRAEnvAdapter",
    snap_at_t: "Snapshot",
    replacement_action: np.ndarray,
    original_actions: np.ndarray,     # (W, chunk_len, action_dim)
    replacement_idx: int,
    role_map: "ObjectRoleMap",
    cfg: "CAPRAConfig",
) -> float:
    """Execute replacement then replay; return total downstream footprint."""
    from experiments.robot.capra.mining.snapshot import restore_snapshot
    from experiments.robot.capra.core.signals import read_state_signals
    from experiments.robot.capra.core.footprint import aggregate_footprint_components, compute_footprint

    H_attr = cfg.attribution_rollout_len
    restore_snapshot(env, snap_at_t)

    total = 0.0
    obs = snap_at_t.obs
    info = snap_at_t.info

    # Phase 1: replacement action
    for i in range(min(len(replacement_action), H_attr)):
        s_b = read_state_signals(obs, step=0, env=env)
        obs, _, done, info = env.step(replacement_action[i])
        s_a = read_state_signals(obs, step=1, env=env, poses_before=s_b.object_poses)
        total += compute_footprint(aggregate_footprint_components(s_b, s_a, role_map), cfg)
        if done:
            return total

    # Phase 2: replay remaining original actions
    for j in range(replacement_idx + 1, len(original_actions)):
        for a in original_actions[j]:
            s_b = read_state_signals(obs, step=0, env=env)
            obs, _, done, info = env.step(a)
            s_a = read_state_signals(obs, step=1, env=env, poses_before=s_b.object_poses)
            total += compute_footprint(aggregate_footprint_components(s_b, s_a, role_map), cfg)
            if done:
                return total
    return total


# ---------------------------------------------------------------------------
# Full env-based attribution
# ---------------------------------------------------------------------------

def compute_precursor_chain(
    env: "CAPRAEnvAdapter",
    trajectory_snaps: List["Snapshot"],
    trajectory_actions: np.ndarray,         # (W, chunk_len, action_dim)
    trajectory_footprints: np.ndarray,      # (W,)
    trajectory_records: List[Any],          # List[TimestepRecord]
    anchor_step: int,
    role_map: "ObjectRoleMap",
    cfg: "CAPRAConfig",
) -> PrecursorChain:
    """Full env-based precursor attribution.

    Uses live env rollouts to measure counterfactual hazard reduction.
    """
    W = len(trajectory_snaps)
    max_steps = min(W, cfg.attribution_max_steps)
    max_reps  = cfg.attribution_max_replacements

    # Suffix hazard baseline
    suffix = np.zeros(W + 1, dtype=np.float64)
    for i in range(W - 1, -1, -1):
        suffix[i] = suffix[i + 1] + float(trajectory_footprints[i])

    # Prioritise steps by descending footprint
    step_order = np.argsort(-trajectory_footprints)[:max_steps]

    raw: List[Tuple[int, float, np.ndarray]] = []

    for idx in step_order:
        rec = trajectory_records[idx]
        snap = trajectory_snaps[idx]

        eq_idx = getattr(rec, 'equivalent_indices', np.array([]))
        fp_vals = getattr(rec, 'footprint_values', np.array([]))

        if len(eq_idx) == 0 and len(fp_vals) > 0:
            eq_idx = np.argsort(fp_vals)[:max_reps]
        elif len(eq_idx) > 0:
            eq_idx = eq_idx[np.argsort(fp_vals[eq_idx])][:max_reps]
        else:
            continue

        best_after = float('inf')
        best_action = trajectory_actions[idx]

        for ci in eq_idx:
            replacement = rec.candidate_actions[ci]
            try:
                h = _replacement_downstream_hazard(
                    env, snap, replacement, trajectory_actions, int(idx), role_map, cfg
                )
            except Exception:
                continue
            if h < best_after:
                best_after = h
                best_action = replacement

        if best_after == float('inf'):
            continue

        delta = max(0.0, float(suffix[idx]) - best_after)
        raw.append((int(idx), delta, best_action))

    if not raw:
        return PrecursorChain(anchor_step=anchor_step, window=W)

    total = sum(d for _, d, _ in raw) + 1e-12
    entries = [
        PrecursorEntry(
            step=anchor_step - W + idx,
            delta_hazard=delta,
            attribution_score=delta / total,
            replacement_action=action,
        )
        for idx, delta, action in sorted(raw, key=lambda x: x[0])
    ]
    return PrecursorChain(anchor_step=anchor_step, window=W, entries=entries)
