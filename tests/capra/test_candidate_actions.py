"""Tests for candidate_actions.py -- pure numpy, no VLA required."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.robot.capra.core.capra_config import CAPRAConfig
from experiments.robot.capra.mining.candidate_actions import (
    uniform_prior_weights,
    synthetic_candidates,
)


# ---------------------------------------------------------------------------
# uniform_prior_weights
# ---------------------------------------------------------------------------

class TestUniformPriorWeights:
    def test_shape(self):
        w = uniform_prior_weights(8)
        assert w.shape == (8,)

    def test_sums_to_one(self):
        for K in (1, 4, 8, 16):
            w = uniform_prior_weights(K)
            assert abs(w.sum() - 1.0) < 1e-6

    def test_all_equal(self):
        w = uniform_prior_weights(5)
        assert np.allclose(w, 1.0 / 5)

    def test_dtype_float32(self):
        w = uniform_prior_weights(4)
        assert w.dtype == np.float32


# ---------------------------------------------------------------------------
# synthetic_candidates (no VLA)
# ---------------------------------------------------------------------------

class TestSyntheticCandidates:
    def test_shapes(self):
        K, chunk_len, action_dim = 8, 8, 7
        actions, prior = synthetic_candidates(K, chunk_len, action_dim)
        assert actions.shape == (K, chunk_len, action_dim)
        assert prior.shape == (K,)

    def test_prior_sums_to_one(self):
        _, prior = synthetic_candidates(4)
        assert abs(prior.sum() - 1.0) < 1e-6

    def test_actions_in_range(self):
        actions, _ = synthetic_candidates(8, 8, 7)
        assert actions.min() >= -1.0 - 1e-6
        assert actions.max() <= 1.0 + 1e-6

    def test_dtype_float32(self):
        actions, _ = synthetic_candidates(4)
        assert actions.dtype == np.float32

    def test_reproducible_with_rng(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        a1, _ = synthetic_candidates(4, rng=rng1)
        a2, _ = synthetic_candidates(4, rng=rng2)
        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds_give_different_actions(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(1)
        a1, _ = synthetic_candidates(4, rng=rng1)
        a2, _ = synthetic_candidates(4, rng=rng2)
        assert not np.allclose(a1, a2)

    def test_k_distinct_candidates(self):
        """With noise, K candidates should not be identical."""
        K = 8
        actions, _ = synthetic_candidates(K)
        # No two rows should be exactly equal
        for i in range(K):
            for j in range(i + 1, K):
                assert not np.allclose(actions[i], actions[j])


# ---------------------------------------------------------------------------
# CAPRAConfig candidate knobs
# ---------------------------------------------------------------------------

class TestCandidateConfig:
    def test_default_K(self):
        cfg = CAPRAConfig()
        assert cfg.K == 8

    def test_default_noise_sigma(self):
        cfg = CAPRAConfig()
        assert cfg.candidate_noise_sigma == pytest.approx(0.02)

    def test_override_K(self):
        cfg = CAPRAConfig(K=16)
        assert cfg.K == 16

    def test_override_H_s(self):
        cfg = CAPRAConfig(H_s=10)
        assert cfg.H_s == 10

    def test_override_noise_sigma(self):
        cfg = CAPRAConfig(candidate_noise_sigma=0.05)
        assert cfg.candidate_noise_sigma == pytest.approx(0.05)
