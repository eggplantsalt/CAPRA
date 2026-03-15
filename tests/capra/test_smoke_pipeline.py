"""Smoke tests: full Phase 3+4 mining pipeline on synthetic state.

Covers:
- candidate generation + equivalence + TimestepRecord (Phase 3)
- mining cache save/load round-trip
- dataset builder build_full_dataset / load_full_dataset
- SAB empty path and non-empty path
- SAB save/load round-trip
"""
from __future__ import annotations

import tempfile
from pathlib import Path

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
from experiments.robot.capra.signals import ObjectPose, StateSignals
from experiments.robot.capra.footprint import aggregate_footprint_components, compute_footprint
from experiments.robot.capra.rollout import TimestepRecord
from experiments.robot.capra.build_capra_dataset import (
    build_safety_target_distribution,
    build_full_dataset,
    load_full_dataset,
)
from experiments.robot.capra.mining_cache import (
    CAPRAEpisodeCache, CAPRATimestepRecord,
    save_episode_cache, load_episode_cache, list_cached_episode_ids,
)
from experiments.robot.capra.buffer import SafetyAlternativeBuffer, BufferEntry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pose(name, pos):
    return ObjectPose(name=name, position=np.array(pos, float),
                      orientation=np.array([0., 0., 0., 1.]))


def _sigs(step, mug_pos, cup_pos):
    return StateSignals(
        step=step,
        object_poses={"mug": _pose("mug", mug_pos),
                      "cup": _pose("cup", cup_pos)},
        contacts=[], support_relations=[],
        topple_flags={"cup": False},
        workspace_violations={"mug": False, "cup": False},
    )


def _cfg(**kw):
    return CAPRAConfig(
        K=6, H_s=5, progress_floor=0.10,
        epsilon_p_abs=0.15, epsilon_p_rel=0.20,
        alpha_d=1.0, alpha_i=0.0, alpha_r=0.0, **kw,
    )


def _make_cache_record(episode_id="ep000", step=5, K=6, force_activate=True):
    """Build a CAPRATimestepRecord; optionally force E_t non-empty."""
    cfg = _cfg()
    rng = np.random.default_rng(42 + step)
    candidate_actions, prior = synthetic_candidates(K, rng=rng)
    role_map = assign_roles_manual(target=["mug"], protected=[], non_target=["cup"])
    base_mug, base_cup = [0., 0., 0.82], [0.1, 0., 0.82]
    progress_values  = np.zeros(K, dtype=np.float32)
    footprint_values = np.zeros(K, dtype=np.float32)
    for i in range(K):
        lift  = max(0., float(candidate_actions[i, :, 2].mean())) * 0.10
        sweep = float(candidate_actions[i, :, 0].mean()) * 0.05
        progress_values[i] = float(np.clip(lift / 0.10, 0., 1.))
        s_b = _sigs(step,     base_mug, base_cup)
        s_a = _sigs(step + 5, np.array(base_mug)+[0,0,max(0,lift)],
                    np.array(base_cup)+[sweep,0,0])
        footprint_values[i] = compute_footprint(
            aggregate_footprint_components(s_b, s_a, role_map), cfg
        )
    _, eq_idx, p_max = build_task_equivalent_set(candidate_actions, progress_values, cfg)
    if force_activate and len(eq_idx) == 0:
        eq_idx = np.array([0], dtype=np.int32)
    eq_idx = eq_idx.astype(np.int32)
    safest_idx = int(local_safest_action_index(eq_idx, footprint_values)) if len(eq_idx) > 0 else -1
    delta_t = compute_local_avoidable_risk(
        float(footprint_values[0]), float(footprint_values[safest_idx])
    ) if safest_idx >= 0 else 0.0
    return CAPRATimestepRecord(
        step=step,
        candidate_actions=candidate_actions, prior_weights=prior,
        progress_values=progress_values, footprint_values=footprint_values,
        equivalent_indices=eq_idx, p_max=p_max, delta_t=delta_t,
        safest_action_idx=safest_idx,
        task_description="pick up the mug", episode_id=episode_id,
    )


# ---------------------------------------------------------------------------
# Phase 3 pipeline (kept)
# ---------------------------------------------------------------------------

class TestShortRolloutMiningSmoke:
    def setup_method(self):
        self.cfg = _cfg()
        self.role_map = assign_roles_manual(target=["mug"], protected=[], non_target=["cup"])
        self.rng = np.random.default_rng(42)

    def test_full_pipeline_nonempty_record(self):
        K = self.cfg.K
        actions, prior = synthetic_candidates(K, rng=self.rng)
        prog = np.zeros(K, dtype=np.float32)
        fp   = np.zeros(K, dtype=np.float32)
        fps  = []
        for i in range(K):
            lift  = max(0., float(actions[i,:,2].mean())) * 0.10
            sweep = float(actions[i,:,0].mean()) * 0.05
            prog[i] = float(np.clip(lift/0.10, 0., 1.))
            s_b = _sigs(10, [0,0,.82], [.1,0,.82])
            s_a = _sigs(15, [0,0,.82+max(0,lift)], [.1+sweep,0,.82])
            c = aggregate_footprint_components(s_b, s_a, self.role_map)
            fp[i] = compute_footprint(c, self.cfg)
            fps.append(c)
        _, eq_idx, p_max = build_task_equivalent_set(actions, prog, self.cfg)
        delta_t = 0.0
        if len(eq_idx) > 0:
            si = local_safest_action_index(eq_idx, fp)
            delta_t = compute_local_avoidable_risk(float(fp[0]), float(fp[si]))
        rec = TimestepRecord(
            episode_id="smoke_ep", step=10,
            candidate_actions=actions, prior_weights=prior,
            progress_values=prog, footprint_values=fp,
            footprint_components=fps,
            equivalent_indices=eq_idx, p_max=p_max, delta_t=delta_t,
        )
        assert rec.candidate_actions.shape == (K, 8, 7)
        assert rec.delta_t >= 0.0
        if len(eq_idx) > 0:
            q = build_safety_target_distribution(fp, eq_idx, prior, beta=1.0)
            assert abs(q.sum() - 1.0) < 1e-5

    def test_e_t_empty_no_progress(self):
        K = self.cfg.K
        actions, _ = synthetic_candidates(K, rng=self.rng)
        _, eq_idx, _ = build_task_equivalent_set(
            actions, np.zeros(K, np.float32), self.cfg
        )
        assert len(eq_idx) == 0

    def test_all_equiv_tight_thresholds(self):
        cfg = CAPRAConfig(progress_floor=0.20, epsilon_p_abs=0.10, epsilon_p_rel=0.20)
        K = 6
        actions = np.random.default_rng(7).standard_normal((K,8,7)).astype(np.float32)
        prog = np.array([.80,.81,.82,.80,.81,.82], dtype=np.float32)
        _, eq_idx, _ = build_task_equivalent_set(actions, prog, cfg)
        assert len(eq_idx) == K

    def test_safest_is_min_fp_in_E_t(self):
        K = 5
        actions = np.random.default_rng(3).standard_normal((K,8,7)).astype(np.float32)
        prog = np.array([.80,.82,.81,.79,.80], dtype=np.float32)
        fp   = np.array([.3,.1,.5,.2,.4], dtype=np.float32)
        cfg = CAPRAConfig(progress_floor=0.20, epsilon_p_abs=0.05, epsilon_p_rel=0.10)
        _, eq_idx, _ = build_task_equivalent_set(actions, prog, cfg)
        safest = local_safest_action_index(eq_idx, fp)
        assert fp[safest] == fp[eq_idx].min()


# ---------------------------------------------------------------------------
# Mining cache round-trip
# ---------------------------------------------------------------------------

class TestMiningCacheRoundTrip:
    def test_save_load_single_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _make_cache_record(episode_id="ep001", step=5)
            cache = CAPRAEpisodeCache(
                episode_id="ep001", task_description="pick up the mug",
                dataset_name="suite_t", total_steps=100,
            )
            cache.append(rec)
            path = save_episode_cache(cache, Path(tmp))
            assert path.exists()
            loaded = load_episode_cache(path)
            assert loaded.episode_id == "ep001"
            assert loaded.n_activated == 1
            r = loaded.records[0]
            assert r.step == 5
            assert r.candidate_actions.shape == rec.candidate_actions.shape
            np.testing.assert_allclose(r.progress_values, rec.progress_values, atol=1e-5)

    def test_save_load_empty_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CAPRAEpisodeCache(
                episode_id="ep_empty", task_description="",
                dataset_name="suite_t", total_steps=10,
            )
            path = save_episode_cache(cache, Path(tmp))
            loaded = load_episode_cache(path)
            assert loaded.n_activated == 0
            assert loaded.records == []

    def test_save_load_multiple_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CAPRAEpisodeCache(
                episode_id="ep002", task_description="pick up the mug",
                dataset_name="suite_t", total_steps=200,
            )
            for step in [10, 30, 50]:
                cache.append(_make_cache_record(episode_id="ep002", step=step))
            path = save_episode_cache(cache, Path(tmp))
            loaded = load_episode_cache(path)
            assert loaded.n_activated == 3
            for i, r in enumerate(loaded.records):
                assert r.step == cache.records[i].step

    def test_list_cached_episode_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            cr = Path(tmp)
            for ep_id in ["ep_a", "ep_b", "ep_c"]:
                c = CAPRAEpisodeCache(
                    episode_id=ep_id, task_description="",
                    dataset_name="my_suite", total_steps=10,
                )
                save_episode_cache(c, cr)
            ids = list_cached_episode_ids(cr, "my_suite")
            assert set(ids) == {"ep_a", "ep_b", "ep_c"}

    def test_resume_skip_logic(self):
        with tempfile.TemporaryDirectory() as tmp:
            cr = Path(tmp)
            c = CAPRAEpisodeCache(
                episode_id="done_ep", task_description="",
                dataset_name="suite", total_steps=5,
            )
            save_episode_cache(c, cr)
            cached = set(list_cached_episode_ids(cr, "suite"))
            assert "done_ep" in cached
            assert "new_ep" not in cached


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class TestDatasetBuilder:
    def test_q_hat_sums_to_one(self):
        fp = np.array([0.1, 0.05, 0.3, 0.08])
        eq = np.array([0, 1, 3])
        prior = np.ones(4) / 4
        q = build_safety_target_distribution(fp, eq, prior, beta=1.0)
        assert abs(q.sum() - 1.0) < 1e-5

    def test_q_hat_zero_outside_E_t(self):
        fp = np.array([0.1, 0.05, 0.3, 0.08])
        eq = np.array([0, 1, 3])
        prior = np.ones(4) / 4
        q = build_safety_target_distribution(fp, eq, prior, beta=1.0)
        assert q[2] == pytest.approx(0.0)

    def test_q_hat_lower_fp_higher_prob(self):
        fp = np.array([0.5, 0.1])
        eq = np.array([0, 1])
        prior = np.ones(2) / 2
        q = build_safety_target_distribution(fp, eq, prior, beta=2.0)
        assert q[1] > q[0]

    def test_q_hat_empty_E_t_all_zero(self):
        fp = np.array([0.1, 0.2])
        eq = np.array([], dtype=int)
        prior = np.ones(2) / 2
        q = build_safety_target_distribution(fp, eq, prior, beta=1.0)
        assert q.sum() == pytest.approx(0.0)

    def test_build_full_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cr = Path(tmp)
            cfg = _cfg()
            for ep_id in ["ep10", "ep11"]:
                rec = _make_cache_record(episode_id=ep_id, step=7)
                cache = CAPRAEpisodeCache(
                    episode_id=ep_id, task_description="pick up the mug",
                    dataset_name="suite_x", total_steps=50,
                )
                cache.append(rec)
                save_episode_cache(cache, cr)
            out = build_full_dataset(cr, "suite_x", cfg)
            assert out.exists()
            data = load_full_dataset(out)
            n = int(data["n_samples"])
            assert n == 2
            assert data["q_hats"].shape[0] == n
            for row in data["q_hats"]:
                assert abs(row.sum() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# SAB -- empty path
# ---------------------------------------------------------------------------

class TestBufferEmptyPath:
    def test_empty_retrieve_returns_empty_list(self):
        buf = SafetyAlternativeBuffer(max_size=100)
        assert len(buf) == 0
        assert buf.retrieve(np.zeros(16, dtype=np.float32), top_k=4) == []

    def test_empty_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = SafetyAlternativeBuffer(max_size=100)
            path = Path(tmp) / "sab.npz"
            buf.save(path)
            buf2 = SafetyAlternativeBuffer.from_file(path)
            assert len(buf2) == 0

    def test_retrieve_top_k_clamped_to_zero(self):
        buf = SafetyAlternativeBuffer(max_size=100)
        assert buf.retrieve(np.zeros(4, np.float32), top_k=999) == []


# ---------------------------------------------------------------------------
# SAB -- non-empty path
# ---------------------------------------------------------------------------

class TestBufferNonEmptyPath:
    def _entry(self, emb, fp=0.1, prog=0.5, ep="ep0", step=0):
        return BufferEntry(
            embedding=np.array(emb, dtype=np.float32),
            action_chunk=np.zeros((8, 7), dtype=np.float32),
            footprint=fp, progress=prog,
            task_description="pick up the mug",
            source_episode=ep, step=step,
        )

    def test_insert_increments_len(self):
        buf = SafetyAlternativeBuffer(max_size=10)
        for i in range(5):
            buf.insert(self._entry([float(i)] * 4))
        assert len(buf) == 5

    def test_retrieve_returns_closest(self):
        buf = SafetyAlternativeBuffer(max_size=50)
        buf.insert(self._entry([0., 0., 0., 0.], fp=0.5))
        buf.insert(self._entry([1., 0., 0., 0.], fp=0.1))
        buf.insert(self._entry([5., 5., 5., 5.], fp=0.8))
        results = buf.retrieve(np.array([0.1, 0., 0., 0.], dtype=np.float32), top_k=1)
        assert len(results) == 1
        np.testing.assert_allclose(results[0].embedding, [0., 0., 0., 0.], atol=1e-5)

    def test_retrieve_top_k(self):
        buf = SafetyAlternativeBuffer(max_size=50)
        for i in range(10):
            buf.insert(self._entry([float(i)] * 4))
        assert len(buf.retrieve(np.zeros(4, np.float32), top_k=3)) == 3

    def test_eviction_at_max_size(self):
        buf = SafetyAlternativeBuffer(max_size=3)
        for i in range(5):
            buf.insert(self._entry([float(i)] * 4, ep=f"ep{i}"))
        assert len(buf) == 3
        eps = {e.source_episode for e in buf._entries}
        assert "ep0" not in eps
        assert "ep4" in eps

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = SafetyAlternativeBuffer(max_size=50)
            for i in range(4):
                buf.insert(self._entry([float(i)] * 8, fp=float(i) * 0.1))
            path = Path(tmp) / "sab.npz"
            buf.save(path)
            buf2 = SafetyAlternativeBuffer.from_file(path)
            assert len(buf2) == 4
            fps_orig = [e.footprint for e in buf._entries]
            fps_load = [e.footprint for e in buf2._entries]
            np.testing.assert_allclose(fps_orig, fps_load, atol=1e-5)

    def test_make_embedding_key_concatenates(self):
        vla_emb = np.ones(8, dtype=np.float32)
        geo = np.array([1., 2., 3.], dtype=np.float32)
        key = SafetyAlternativeBuffer.make_embedding_key(vla_emb, geo)
        assert key.shape == (11,)
        np.testing.assert_allclose(key[8:], geo)

    def test_make_embedding_key_no_geo(self):
        key = SafetyAlternativeBuffer.make_embedding_key(np.ones(8, np.float32), None)
        assert key.shape == (8,)
