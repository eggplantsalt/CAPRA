"""Smoke test: verify all CAPRA module imports and pure-Python logic.

Does NOT require a GPU, model checkpoint, or LIBERO installation.
All env-dependent functions are expected to raise NotImplementedError
(Phase 2 stubs) -- that is the correct behaviour here.
"""
import numpy as np
import pytest


def test_imports():
    """All CAPRA modules must import without error."""
    from experiments.robot.capra import capra_config
    from experiments.robot.capra import env_adapter
    from experiments.robot.capra import snapshot
    from experiments.robot.capra import state_api
    from experiments.robot.capra import object_roles
    from experiments.robot.capra import task_progress
    from experiments.robot.capra import signals
    from experiments.robot.capra import footprint
    from experiments.robot.capra import equivalence
    from experiments.robot.capra import candidate_actions
    from experiments.robot.capra import buffer
    from experiments.robot.capra import rollout
    from experiments.robot.capra import precursor
    from experiments.robot.capra import mining_cache
    from experiments.robot.capra import build_capra_dataset
    from experiments.robot.capra import metrics
    from experiments.robot.capra import procedural_splits
    from experiments.robot.capra import report_utils


def test_config_defaults():
    from experiments.robot.capra.capra_config import CAPRAConfig
    cfg = CAPRAConfig()
    assert cfg.K == 8
    assert cfg.H_s == 5
    assert cfg.shuffle_buffer_size == 2000
    assert cfg.lam == pytest.approx(0.1)


def test_equivalence_filter_smoke():
    from experiments.robot.capra.capra_config import CAPRAConfig
    from experiments.robot.capra.equivalence import build_task_equivalent_set
    cfg = CAPRAConfig()
    actions  = np.random.randn(4, 8, 7).astype(np.float32)
    progress = np.array([0.80, 0.82, 0.60, 0.81])
    eq_actions, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
    assert p_max == pytest.approx(0.82)
    assert len(eq_idx) >= 1


def test_safety_target_distribution_smoke():
    from experiments.robot.capra.build_capra_dataset import build_safety_target_distribution
    footprints = np.array([0.1, 0.05, 0.3, 0.08])
    eq_idx     = np.array([0, 1, 3])
    prior      = np.ones(4) / 4
    q_hat = build_safety_target_distribution(footprints, eq_idx, prior, beta=1.0)
    assert abs(q_hat.sum() - 1.0) < 1e-5


def test_metrics_smoke():
    from experiments.robot.capra.metrics import compute_spir, compute_ear
    chosen  = np.array([0.3, 0.1, 0.4])
    min_eq  = np.array([0.1, 0.1, 0.1])
    active  = np.array([True, True, True])
    spir = compute_spir(chosen, min_eq, active)
    ear  = compute_ear(chosen - min_eq, active)
    assert 0.0 <= spir <= 1.0
    assert ear >= 0.0


def test_phase2_stubs_raise():
    """Phase 2 stubs must raise NotImplementedError, not other errors."""
    from experiments.robot.capra.snapshot import save_snapshot, restore_snapshot
    from experiments.robot.capra.state_api import read_state_signals
    with pytest.raises(NotImplementedError):
        save_snapshot(None)  # type: ignore
    with pytest.raises(NotImplementedError):
        read_state_signals(None, {}, 0)  # type: ignore
