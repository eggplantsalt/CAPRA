"""Smoke test: run the full short-rollout mining pipeline on a synthetic state.

This test exercises the entire Phase 3 pipeline WITHOUT a live env or VLA:
  candidate_actions.synthetic_candidates
  -> equivalence.build_task_equivalent_set
  -> equivalence.local_safest_action_index
  -> equivalence.compute_local_avoidable_risk
  -> rollout.TimestepRecord construction
  -> build_capra_dataset.build_safety_target_distribution

A MockEnv simulates H_s env steps by returning pre-scripted observations
that encode:
  - task progress increasing for actions that move the target object up
  - non-zero footprint for actions that displace non-target objects

The test asserts that the pipeline produces a non-empty TimestepRecord
with coherent fields.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.robot.capra.capra_config import CAPRAConfig
from experiments.robot.capra.candidate_actions import synthetic_candidates, uniform_prior_weights
from experiments.robot.capra.equivalence import (
    build_task_equivalent_set,
    local_safest_action_index,
    compute_local_avoidable_risk,
)
from experiments.robot.capra.object_roles import assign_roles_manual
from experiments.robot.capra.signals import (
    ObjectPose, ContactEvent, StateSignals,
)
from experiments.robot.capra.footprint import (
    aggregate_footprint_components, compute_footprint,
)
from experiments.robot.capra.task_progress import compute_progress_from_rollout
from experiments.robot.capra.rollout import TimestepRecord
from experiments.robot.capra.build_capra_dataset import build_safety_target_distribution


# ---------------------------------------------------------------------------
# Synthetic "env" that scripted per-candidate observations
# ---------------------------------------------------------------------------

def _make_pose(name, pos, quat=None):
    if quat is None:
        quat = np.array([0., 0., 0., 1.])
    return ObjectPose(name=name, position=np.array(pos, float),
                      orientation=np.array(quat, float))


def _make_signals(step, mug_pos, cup_pos, topple=False):
    """Build a minimal StateSignals for the smoke test."""
    return StateSignals(
        step=step,
        object_poses={
            "mug": _make_pose("mug", mug_pos),
            "cup": _make_pose("cup", cup_pos),
        },
        contacts=[],
        support_relations=[],
        topple_flags={"cup": topple},
        workspace_violations={"mug": False, "cup": False},
    )


def _scripted_rollout(action_chunk, base_mug_pos, base_cup_pos, step_offset):
    """Simulate a short rollout: deterministic outcome from action_chunk[0,2].

    - Action dim 2 (z) controls mug lift (progress).
    - Action dim 0 (x) controls cup displacement (footprint).
    Returns (progress, footprint_components, signals_before, signals_after).
    """
    # Aggregate effect over H_s=5 steps
    lift  = float(np.clip(action_chunk[:, 2].mean(), -1, 1)) * 0.10  # up to 10cm lift
    sweep = float(np.clip(action_chunk[:, 0].mean(), -1, 1)) * 0.05  # up to 5cm sweep

    mug_after = np.array(base_mug_pos) + np.array([0., 0., max(0., lift)])
    cup_after = np.array(base_cup_pos) + np.array([sweep, 0., 0.])

    signals_before = _make_signals(step_offset, base_mug_pos, base_cup_pos)
    signals_after  = _make_signals(step_offset + 5, mug_after, cup_after)

    return signals_before, signals_after


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestShortRolloutMiningSmoke:
    def setup_method(self):
        self.cfg = CAPRAConfig(
            K=8, H_s=5,
            progress_floor=0.10,
            epsilon_p_abs=0.15,
            epsilon_p_rel=0.20,
            alpha_d=1.0, alpha_i=0.0, alpha_r=0.0,  # displacement only for clarity
        )
        self.role_map = assign_roles_manual(
            target=["mug"],
            protected=[],
            non_target=["cup"],
        )
        self.rng = np.random.default_rng(42)

    def test_full_pipeline_produces_nonempty_record(self):
        """Run the complete Phase 3 pipeline and assert non-empty output."""
        K = self.cfg.K
        base_mug = [0.0, 0.0, 0.82]
        base_cup = [0.1, 0.0, 0.82]

        # 1. Generate K synthetic candidates
        candidate_actions, prior_weights = synthetic_candidates(
            K, chunk_len=8, action_dim=7, rng=self.rng
        )
        assert candidate_actions.shape == (K, 8, 7)

        # 2. Run scripted rollout for each candidate
        progress_values  = np.zeros(K, dtype=np.float32)
        footprint_values = np.zeros(K, dtype=np.float32)
        fp_components    = []

        for i in range(K):
            sig_before, sig_after = _scripted_rollout(
                candidate_actions[i], base_mug, base_cup, step_offset=10
            )
            comps  = aggregate_footprint_components(sig_before, sig_after, self.role_map)
            fp_val = compute_footprint(comps, self.cfg)
            footprint_values[i] = fp_val
            fp_components.append(comps)

            # Synthetic progress: proportional to mug lift
            lift = max(0., candidate_actions[i, :, 2].mean()) * 0.10
            progress_values[i] = float(np.clip(lift / 0.10, 0., 1.))

        # 3. Build equivalence set
        eq_actions, eq_idx, p_max = build_task_equivalent_set(
            candidate_actions, progress_values, self.cfg
        )

        # With random actions some should lift the mug > progress_floor
        # Assert p_max is computed
        assert p_max == pytest.approx(float(progress_values.max()))

        # 4. If E_t is non-empty, compute Delta_t
        if len(eq_idx) > 0:
            safest_idx = local_safest_action_index(eq_idx, footprint_values)
            delta_t = compute_local_avoidable_risk(
                chosen_footprint=float(footprint_values[0]),
                min_equivalent_footprint=float(footprint_values[safest_idx]),
            )
            assert delta_t >= 0.0
        else:
            delta_t = 0.0

        # 5. Build TimestepRecord
        record = TimestepRecord(
            episode_id="smoke_ep_000",
            step=10,
            candidate_actions=candidate_actions,
            prior_weights=prior_weights,
            progress_values=progress_values,
            footprint_values=footprint_values,
            footprint_components=fp_components,
            equivalent_indices=eq_idx,
            p_max=p_max,
            delta_t=delta_t,
            task_description="pick up the mug",
        )

        # 6. Assertions: record is non-empty and internally consistent
        assert record.episode_id == "smoke_ep_000"
        assert record.step == 10
        assert record.candidate_actions.shape == (K, 8, 7)
        assert len(record.progress_values) == K
        assert len(record.footprint_values) == K
        assert len(record.footprint_components) == K
        assert record.p_max == pytest.approx(p_max)
        assert record.delta_t >= 0.0
        assert len(record.prior_weights) == K
        assert abs(record.prior_weights.sum() - 1.0) < 1e-5

        # 7. Safety target distribution (if E_t non-empty)
        if len(eq_idx) > 0:
            q_hat = build_safety_target_distribution(
                record.footprint_values,
                record.equivalent_indices,
                record.prior_weights,
                beta=self.cfg.beta,
            )
            assert q_hat.shape == (K,)
            assert abs(q_hat.sum() - 1.0) < 1e-5
            # Only equivalent candidates have non-zero probability
            for i in range(K):
                if i not in eq_idx:
                    assert q_hat[i] == pytest.approx(0.0)

    def test_e_t_empty_when_no_progress(self):
        """When all candidates make no progress, E_t must be empty."""
        K = self.cfg.K
        candidate_actions, prior_weights = synthetic_candidates(K, rng=self.rng)
        # Force all progress to 0
        progress_values = np.zeros(K, dtype=np.float32)
        eq_a, eq_idx, p_max = build_task_equivalent_set(
            candidate_actions, progress_values, self.cfg
        )
        assert len(eq_idx) == 0
        assert p_max == pytest.approx(0.0)

    def test_all_candidates_equivalent_when_tight_thresholds(self):
        """When all progress values are nearly identical and above floor."""
        cfg = CAPRAConfig(
            progress_floor=0.20,
            epsilon_p_abs=0.10,
            epsilon_p_rel=0.20,
        )
        K = 6
        candidate_actions = np.random.default_rng(7).standard_normal((K, 8, 7)).astype(np.float32)
        # All progress values in [0.80, 0.82] -- all within epsilon_p_abs=0.10
        progress_values = np.array([0.80, 0.81, 0.82, 0.80, 0.81, 0.82], dtype=np.float32)
        eq_a, eq_idx, p_max = build_task_equivalent_set(candidate_actions, progress_values, cfg)
        assert len(eq_idx) == K

    def test_safest_candidate_is_min_footprint_in_E_t(self):
        """local_safest_action_index returns minimum footprint among E_t."""
        K = 5
        candidate_actions = np.random.default_rng(3).standard_normal((K, 8, 7)).astype(np.float32)
        progress_values   = np.array([0.80, 0.82, 0.81, 0.79, 0.80], dtype=np.float32)
        footprint_values  = np.array([0.3,  0.1,  0.5,  0.2,  0.4],  dtype=np.float32)
        cfg = CAPRAConfig(progress_floor=0.20, epsilon_p_abs=0.05, epsilon_p_rel=0.10)
        _, eq_idx, _ = build_task_equivalent_set(candidate_actions, progress_values, cfg)
        safest = local_safest_action_index(eq_idx, footprint_values)
        # safest must have the minimum footprint among eq_idx members
        assert footprint_values[safest] == footprint_values[eq_idx].min()
