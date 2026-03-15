"""Tests for equivalence.py -- pure Python / numpy, no env required."""
import numpy as np
import pytest

from experiments.robot.capra.capra_config import CAPRAConfig
from experiments.robot.capra.equivalence import (
    build_task_equivalent_set,
    compute_local_avoidable_risk,
    local_safest_action_index,
)


@pytest.fixture
def cfg():
    return CAPRAConfig(progress_floor=0.2, epsilon_p_abs=0.05, epsilon_p_rel=0.10)


def test_all_equivalent(cfg):
    actions = np.random.randn(4, 8, 7).astype(np.float32)
    progress = np.array([0.80, 0.82, 0.81, 0.79])
    eq_actions, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
    assert len(eq_idx) > 0
    assert p_max == pytest.approx(0.82)


def test_below_progress_floor(cfg):
    actions = np.random.randn(4, 8, 7).astype(np.float32)
    progress = np.array([0.10, 0.11, 0.12, 0.09])
    eq_actions, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
    assert len(eq_idx) == 0


def test_only_one_equivalent(cfg):
    actions = np.random.randn(4, 8, 7).astype(np.float32)
    progress = np.array([0.90, 0.50, 0.40, 0.30])
    eq_actions, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
    assert list(eq_idx) == [0]


def test_local_avoidable_risk_non_negative():
    delta = compute_local_avoidable_risk(0.3, 0.1)
    assert delta == pytest.approx(0.2)
    delta_zero = compute_local_avoidable_risk(0.1, 0.3)
    assert delta_zero == pytest.approx(0.0)


def test_safest_action_index(cfg):
    footprints = np.array([0.5, 0.1, 0.3, 0.2])
    eq_idx = np.array([0, 1, 3])
    best = local_safest_action_index(eq_idx, footprints)
    assert best == 1  # footprints[1] = 0.1 is smallest among {0,1,3}
