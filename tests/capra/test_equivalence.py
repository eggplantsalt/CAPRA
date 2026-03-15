"""Tests for equivalence.py -- pure numpy, no env required.

Covers:
- All three gates (progress_floor, abs_gap, rel_gap) work independently.
- local_safest_action_index picks the minimum-footprint member.
- compute_local_avoidable_risk is non-negative and correct.
- Edge cases: empty E_t, single-candidate E_t, all-equivalent set.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.robot.capra.core.capra_config import CAPRAConfig
from experiments.robot.capra.core.equivalence import (
    build_task_equivalent_set,
    local_safest_action_index,
    compute_local_avoidable_risk,
)


@pytest.fixture
def cfg():
    return CAPRAConfig(
        progress_floor=0.20,
        epsilon_p_abs=0.05,
        epsilon_p_rel=0.10,
    )


def make_actions(K: int, chunk_len: int = 8, action_dim: int = 7) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((K, chunk_len, action_dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Gate 1: progress_floor
# ---------------------------------------------------------------------------

class TestProgressFloor:
    def test_all_below_floor_returns_empty(self, cfg):
        actions   = make_actions(4)
        progress  = np.array([0.05, 0.10, 0.12, 0.08])  # all < 0.20
        eq_a, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
        assert len(eq_idx) == 0
        assert eq_a.shape[0] == 0
        assert p_max == pytest.approx(0.12)

    def test_p_max_exactly_at_floor_empty(self, cfg):
        actions  = make_actions(3)
        progress = np.array([0.20, 0.15, 0.18])  # p_max == floor
        # p_max=0.20 passes floor gate, abs_gap for [0]=0 passes,
        # so index 0 should be included
        eq_a, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
        assert 0 in eq_idx

    def test_p_max_above_floor_nonempty(self, cfg):
        actions  = make_actions(4)
        progress = np.array([0.80, 0.82, 0.60, 0.81])
        eq_a, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
        assert len(eq_idx) > 0


# ---------------------------------------------------------------------------
# Gate 2: absolute progress gap
# ---------------------------------------------------------------------------

class TestAbsoluteGap:
    def test_large_abs_gap_excluded(self, cfg):
        actions  = make_actions(3)
        # p_max = 0.90; indices 1,2 have gaps 0.30 and 0.40 > epsilon_p_abs=0.05
        progress = np.array([0.90, 0.60, 0.50])
        eq_a, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
        assert list(eq_idx) == [0]

    def test_within_abs_gap_included(self, cfg):
        actions  = make_actions(4)
        # all within 0.04 of p_max=0.85 -- all should pass abs gate
        progress = np.array([0.85, 0.83, 0.81, 0.82])
        eq_a, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
        assert set(eq_idx) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Gate 3: relative progress gap
# ---------------------------------------------------------------------------

class TestRelativeGap:
    def test_rel_gap_excludes_distant(self):
        # Use small epsilon_p_rel to test
        cfg_tight = CAPRAConfig(
            progress_floor=0.20,
            epsilon_p_abs=1.0,   # disable abs gate
            epsilon_p_rel=0.05,  # 5% relative
        )
        actions  = make_actions(3)
        # p_max=0.80; index 1 has rel_gap = 0.20/0.80 = 0.25 > 0.05
        progress = np.array([0.80, 0.60, 0.79])
        eq_a, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg_tight)
        assert 1 not in eq_idx
        assert 0 in eq_idx
        assert 2 in eq_idx


# ---------------------------------------------------------------------------
# local_safest_action_index
# ---------------------------------------------------------------------------

class TestLocalSafestIndex:
    def test_picks_minimum_footprint(self, cfg):
        footprints = np.array([0.5, 0.1, 0.3, 0.2])
        eq_idx     = np.array([0, 1, 3])   # candidates at indices 0,1,3
        best = local_safest_action_index(eq_idx, footprints)
        assert best == 1   # footprints[1]=0.1 is smallest

    def test_single_member(self, cfg):
        footprints = np.array([0.5, 0.1, 0.3])
        eq_idx     = np.array([2])
        best = local_safest_action_index(eq_idx, footprints)
        assert best == 2

    def test_empty_raises(self, cfg):
        with pytest.raises(ValueError):
            local_safest_action_index(np.array([], dtype=int), np.array([0.1]))


# ---------------------------------------------------------------------------
# compute_local_avoidable_risk
# ---------------------------------------------------------------------------

class TestLocalAvoidableRisk:
    def test_positive_delta(self):
        delta = compute_local_avoidable_risk(0.3, 0.1)
        assert delta == pytest.approx(0.2)

    def test_chosen_safer_zero_delta(self):
        delta = compute_local_avoidable_risk(0.1, 0.3)
        assert delta == pytest.approx(0.0)

    def test_equal_footprints_zero_delta(self):
        delta = compute_local_avoidable_risk(0.2, 0.2)
        assert delta == pytest.approx(0.0)

    def test_non_negative(self):
        for chosen, safest in [(0.1, 0.5), (0.0, 1.0), (0.5, 0.5)]:
            assert compute_local_avoidable_risk(chosen, safest) >= 0.0


# ---------------------------------------------------------------------------
# p_max returned correctly
# ---------------------------------------------------------------------------

class TestPMax:
    def test_p_max_is_max_of_progress(self, cfg):
        actions  = make_actions(5)
        progress = np.array([0.3, 0.7, 0.5, 0.9, 0.6])
        _, _, p_max = build_task_equivalent_set(actions, progress, cfg)
        assert p_max == pytest.approx(0.9)

    def test_output_shapes_consistent(self, cfg):
        K = 6
        actions  = make_actions(K)
        progress = np.array([0.80, 0.82, 0.60, 0.81, 0.79, 0.50])
        eq_a, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
        assert eq_a.shape[0] == len(eq_idx)
        assert eq_a.shape[1:] == actions.shape[1:]
