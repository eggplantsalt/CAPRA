"""
任务等价集过滤器 (equivalence.py)
==================================

核心思想
--------
CAPRA 的安全约束只在"有更安全的选择"时才触发。
如果所有候选动作都差不多糟糕，强迫模型选择"最安全"的并不合理。

因此，我们先筛选出"任务等价集" E_t：
  E_t = { a_i | P_t(a_i) 足够接近 P_max 且 >= progress_floor }

只有当 E_t 非空时，才比较候选动作的 footprint，触发 CAPRA 损失。

数学定义
--------
a_i ∈ E_t  iff 所有条件同时满足：
  1) P_t(a_i) >= progress_floor           （最低进度门槛）
  2) |P_max - P_t(a_i)| <= epsilon_p_abs  （绝对进度差）
  3) |P_max - P_t(a_i)| / P_max <= epsilon_p_rel  （相对进度差）

其中 P_max = max_{i} P_t(a_i)。

为什么需要两个阈值（abs + rel）？
  - 绝对阈值防止高进度任务被过度限制
  - 相对阈值防止低进度任务被意外包含
  - 两个条件都满足才算等价，取交集更保守
"""
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.core.capra_config import CAPRAConfig


def build_task_equivalent_set(
    candidate_actions: np.ndarray,   # (K, chunk_len, action_dim) K个候选动作块
    progress_values: np.ndarray,     # (K,)  每个候选动作的任务进度 P_t(a_i)
    cfg: "CAPRAConfig",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    构建任务等价集 E_t。

    返回:
        equivalent_actions: 等价集中的动作块，shape (|E_t|, chunk_len, action_dim)
        equivalent_indices: 等价集中的候选索引，shape (|E_t|,)
        P_max: 所有候选中的最大进度值

    当 E_t 为空时（CAPRA 损失不触发），返回空数组。

    注意：
    - 当 P_max < progress_floor 时，说明任务还没有实质性进展，
      此时不应强加安全约束，直接返回空集。
    - K 个候选中 index=0 始终是 nominal（policy 实际预测的）动作。
    """
    K = len(progress_values)
    assert candidate_actions.shape[0] == K, "actions 和 progress 长度必须一致"

    # 找到所有候选中的最高进度
    P_max = float(np.max(progress_values))

    # 门槛 1：任务进度太低，不触发 CAPRA 损失
    if P_max < cfg.progress_floor:
        empty = np.empty((0,) + candidate_actions.shape[1:], dtype=candidate_actions.dtype)
        return empty, np.array([], dtype=int), P_max

    # 门槛 2 & 3：绝对+相对进度差双重过滤
    abs_gap = np.abs(P_max - progress_values)   # 绝对差
    rel_gap = abs_gap / (P_max + 1e-8)          # 相对差（避免除零）

    # 三个条件取交集
    mask = (
        (progress_values >= cfg.progress_floor)   # 候选自身进度足够
        & (abs_gap <= cfg.epsilon_p_abs)           # 绝对差够小
        & (rel_gap <= cfg.epsilon_p_rel)           # 相对差够小
    )

    indices = np.where(mask)[0]
    return candidate_actions[indices], indices, P_max


def local_safest_action_index(
    equivalent_indices: np.ndarray,  # E_t 中候选的原始索引
    footprint_values: np.ndarray,    # (K,)  所有候选的 footprint F_t(a_i)
) -> int:
    """
    在任务等价集 E_t 中找到 footprint 最小的候选动作索引。

    返回的是原始 K 候选数组中的索引（不是 E_t 的局部索引）。
    这个动作就是"最安全的任务等价选择"，用于计算 Delta_t。

    调用前请确保 equivalent_indices 非空（即 E_t 非空）。
    """
    if len(equivalent_indices) == 0:
        raise ValueError("equivalent_indices 为空 -- CAPRA 损失不应在此触发")
    # 在等价集内找 footprint 最小者
    best_local = int(np.argmin(footprint_values[equivalent_indices]))
    return int(equivalent_indices[best_local])


def compute_local_avoidable_risk(
    chosen_footprint: float,
    min_equivalent_footprint: float,
) -> float:
    """
    计算局部可避免风险 Delta_t。

    Delta_t = F_t(chosen) - F_t(safest_equivalent)

    物理含义：
    - Delta_t > 0：模型选择了一个比等价集中最安全选项代价更高的动作，
                   意味着存在"可避免的副作用"。
    - Delta_t = 0：模型已经选择了最安全的等价动作，或 E_t 为空。
    - 截断到 0 以上，因为如果 nominal 动作比"最安全"还好，Delta_t 为 0 即可。
    """
    return max(0.0, chosen_footprint - min_equivalent_footprint)
