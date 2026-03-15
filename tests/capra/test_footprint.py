"""Tests for build_capra_dataset.build_safety_target_distribution."""
import numpy as np
import pytest

from experiments.robot.capra.build_capra_dataset import build_safety_target_distribution


def test_q_hat_sums_to_one():
    footprints = np.array([0.1, 0.05, 0.3, 0.08])
    eq_idx     = np.array([0, 1, 3])
    prior      = np.ones(4) / 4
    q_hat = build_safety_target_distribution(footprints, eq_idx, prior, beta=1.0)
    assert abs(q_hat.sum() - 1.0) < 1e-5


def test_q_hat_zero_outside_equivalent_set():
    footprints = np.array([0.1, 0.05, 0.3, 0.08])
    eq_idx     = np.array([0, 1, 3])   # index 2 is NOT in E_t
    prior      = np.ones(4) / 4
    q_hat = build_safety_target_distribution(footprints, eq_idx, prior, beta=1.0)
    assert q_hat[2] == pytest.approx(0.0)


def test_q_hat_lower_footprint_gets_higher_prob():
    footprints = np.array([0.5, 0.1])  # index 1 is safer
    eq_idx     = np.array([0, 1])
    prior      = np.ones(2) / 2
    q_hat = build_safety_target_distribution(footprints, eq_idx, prior, beta=2.0)
    assert q_hat[1] > q_hat[0]


def test_q_hat_empty_equivalent_set():
    footprints = np.array([0.1, 0.2])
    eq_idx     = np.array([], dtype=int)
    prior      = np.ones(2) / 2
    q_hat = build_safety_target_distribution(footprints, eq_idx, prior, beta=1.0)
    assert q_hat.sum() == pytest.approx(0.0)
