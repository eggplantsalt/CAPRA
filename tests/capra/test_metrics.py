"""Tests for metrics.py -- pure Python / numpy, no env required."""
import numpy as np
import pytest

from experiments.robot.capra.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    aggregate_episode_metrics,
    compute_attribution_edit_gain,
    compute_ear,
    compute_spir,
)


def test_spir_all_inversions():
    chosen  = np.array([0.3, 0.4, 0.5])
    min_eq  = np.array([0.1, 0.1, 0.1])
    active  = np.array([True, True, True])
    assert compute_spir(chosen, min_eq, active) == pytest.approx(1.0)


def test_spir_no_inversions():
    chosen  = np.array([0.1, 0.1, 0.1])
    min_eq  = np.array([0.3, 0.3, 0.3])
    active  = np.array([True, True, True])
    assert compute_spir(chosen, min_eq, active) == pytest.approx(0.0)


def test_spir_empty_activated():
    chosen  = np.array([0.3, 0.4])
    min_eq  = np.array([0.1, 0.1])
    active  = np.array([False, False])
    assert compute_spir(chosen, min_eq, active) == pytest.approx(0.0)


def test_ear_basic():
    deltas = np.array([0.2, 0.4, 0.0])
    active = np.array([True, True, True])
    assert compute_ear(deltas, active) == pytest.approx(0.2)


def test_ear_partial_activation():
    deltas = np.array([0.0, 0.3, 0.0])
    active = np.array([False, True, False])
    assert compute_ear(deltas, active) == pytest.approx(0.3)


def test_attribution_edit_gain():
    gain = compute_attribution_edit_gain(hazard_before=0.8, hazard_after=0.4)
    assert gain == pytest.approx(0.5)


def test_aggregate_empty():
    result = aggregate_episode_metrics([])
    assert result.n_episodes == 0
    assert result.success_rate == 0.0


def test_aggregate_basic():
    episodes = [
        EpisodeMetrics(success=True,  spir=0.2, ear=0.1),
        EpisodeMetrics(success=False, spir=0.6, ear=0.3),
    ]
    agg = aggregate_episode_metrics(episodes)
    assert agg.n_episodes == 2
    assert agg.success_rate == pytest.approx(0.5)
    assert agg.spir_mean   == pytest.approx(0.4)
    assert agg.ear_mean    == pytest.approx(0.2)
