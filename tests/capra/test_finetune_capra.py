"""Smoke tests for finetune_capra.py -- pure Python, no GPU required.

Tests verify:
  a) baseline mode config and forward pass (capra_enabled=False)
  b) CAPRA mode config and forward pass (capra_enabled=True)
  c) anchor-only branch (warmup suppresses CAPRA, or empty records)
  d) CAPRA-active branch (records present, warmup passed)

All VLA/GPU calls are monkeypatched so these run on CPU without any model.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.robot.capra.core.capra_loss import (
    CAPRADatasetReader,
    compute_capra_kl_loss,
    compute_pi_theta,
    kl_q_hat_pi,
)
from vla_scripts.finetune_capra import FinetuneCAPRAConfig, run_capra_forward_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(K=4, CL=8, A=7, w=0.2, dt=0.1):
    rng = np.random.default_rng(0)
    q = np.zeros(K, dtype=np.float32)
    q[:2] = 0.5
    return {
        "embedding":  np.zeros(8, dtype=np.float32),
        "q_hat":      q,
        "weight":     np.float32(w),
        "actions":    rng.standard_normal((K, CL, A)).astype(np.float32),
        "step":       np.int32(5),
        "episode_id": "ep0",
        "delta_t":    np.float32(dt),
        "p_max":      np.float32(0.8),
    }


DEV = torch.device("cpu")


def _baseline_cfg(**kw):
    return FinetuneCAPRAConfig(
        capra_enabled=False, use_l1_regression=True, use_diffusion=False,
        use_proprio=False, use_film=False, **kw,
    )


def _capra_cfg(**kw):
    return FinetuneCAPRAConfig(
        capra_enabled=True, use_l1_regression=True, use_diffusion=False,
        use_proprio=False, use_film=False,
        lam=0.1, rho=0.5, capra_gamma=0.0, capra_warmup_steps=0, **kw,
    )


def _fake_batch():
    return {
        "input_ids":      torch.zeros(2, 10, dtype=torch.long),
        "attention_mask": torch.ones(2, 10),
        "pixel_values":   torch.zeros(2, 3, 224, 224),
        "labels":         torch.zeros(2, 10, dtype=torch.long),
        "actions":        torch.zeros(2, 8, 7),
    }


# ---------------------------------------------------------------------------
# A) Config tests
# ---------------------------------------------------------------------------

class TestBaselineConfig:
    def test_capra_off_by_default(self):   assert FinetuneCAPRAConfig().capra_enabled is False
    def test_default_lam(self):            assert FinetuneCAPRAConfig().lam == pytest.approx(0.1)
    def test_default_rho(self):            assert FinetuneCAPRAConfig().rho == pytest.approx(0.5)
    def test_default_beta(self):           assert FinetuneCAPRAConfig().beta == pytest.approx(1.0)
    def test_default_warmup(self):         assert FinetuneCAPRAConfig().capra_warmup_steps == 1000
    def test_default_gamma(self):          assert FinetuneCAPRAConfig().capra_gamma == pytest.approx(1.0)
    def test_has_use_lora(self):           assert hasattr(FinetuneCAPRAConfig(), "use_lora")
    def test_has_l1(self):                 assert hasattr(FinetuneCAPRAConfig(), "use_l1_regression")
    def test_shuffle_buffer_2000(self):    assert FinetuneCAPRAConfig().shuffle_buffer_size == 2000


class TestCAPRAConfig:
    def test_enable(self):        assert FinetuneCAPRAConfig(capra_enabled=True).capra_enabled is True
    def test_override_lam(self):  assert FinetuneCAPRAConfig(lam=0.05).lam == pytest.approx(0.05)
    def test_override_warmup(self): assert FinetuneCAPRAConfig(capra_warmup_steps=200).capra_warmup_steps == 200
    def test_override_cache(self):  assert FinetuneCAPRAConfig(cache_root=Path("x")).cache_root == Path("x")
    def test_override_beta(self):   assert FinetuneCAPRAConfig(beta=2.0).beta == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# B) compute_capra_kl_loss
# ---------------------------------------------------------------------------

class TestLossEmpty:
    def test_zero_loss(self):        assert compute_capra_kl_loss([], torch.zeros(4,8,7), DEV)[0].item() == pytest.approx(0.0)
    def test_zero_ratio(self):       assert compute_capra_kl_loss([], torch.zeros(4,8,7), DEV)[1]["activation_ratio"] == pytest.approx(0.0)
    def test_no_grad(self):          assert not compute_capra_kl_loss([], torch.zeros(4,8,7), DEV)[0].requires_grad
    def test_required_keys(self):
        _, m = compute_capra_kl_loss([], torch.zeros(4,8,7), DEV)
        for k in ["capra_loss","activation_ratio","mean_w_t","mean_delta_t"]: assert k in m


class TestLossActivated:
    def test_nonneg(self):
        loss,_ = compute_capra_kl_loss([_record()]*2, torch.randn(4,8,7), DEV)
        assert loss.item() >= 0.0

    def test_ratio_positive(self):
        _,m = compute_capra_kl_loss([_record(w=0.3)], torch.randn(4,8,7), DEV)
        assert m["activation_ratio"] > 0.0

    def test_mean_w_t(self):
        _,m = compute_capra_kl_loss([_record(w=0.25)], torch.randn(4,8,7), DEV)
        assert m["mean_w_t"] == pytest.approx(0.25, abs=1e-4)

    def test_zero_q_hat_zero_loss(self):
        r = _record(); r["q_hat"] = np.zeros(4, np.float32)
        loss,m = compute_capra_kl_loss([r], torch.randn(4,8,7), DEV)
        assert loss.item() == pytest.approx(0.0)
        assert m["activation_ratio"] == pytest.approx(0.0)

    def test_keys_present(self):
        _,m = compute_capra_kl_loss([_record()], torch.randn(4,8,7), DEV)
        for k in ["capra_loss","activation_ratio","mean_w_t","mean_delta_t"]: assert k in m


# ---------------------------------------------------------------------------
# C) compute_pi_theta
# ---------------------------------------------------------------------------

class TestPiTheta:
    def test_uniform_gamma0(self):
        pi = compute_pi_theta(torch.randn(4,8,7), torch.randn(8,7), gamma=0.0)
        assert torch.allclose(pi, torch.ones(4)/4, atol=1e-5)

    def test_sums_to_one(self):
        pi = compute_pi_theta(torch.randn(6,8,7), torch.randn(8,7), gamma=1.0)
        assert abs(pi.sum().item()-1.0) < 1e-5

    def test_closest_highest(self):
        pred  = torch.zeros(4, 3)
        cands = torch.stack([torch.zeros(4,3), torch.ones(4,3)])
        pi = compute_pi_theta(cands, pred, gamma=2.0)
        assert pi[0] > pi[1]

    def test_float32(self):
        assert compute_pi_theta(torch.randn(4,8,7), torch.randn(8,7)).dtype == torch.float32


# ---------------------------------------------------------------------------
# D) kl_q_hat_pi
# ---------------------------------------------------------------------------

class TestKL:
    def test_zero_equal(self):
        q = torch.tensor([0.5,0.3,0.2])
        assert kl_q_hat_pi(q,q.clone()).item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_different(self):
        assert kl_q_hat_pi(torch.tensor([0.8,0.1,0.1]), torch.tensor([0.3,0.4,0.3])).item() > 0.0

    def test_asymmetric(self):
        q,p = torch.tensor([0.9,0.1]), torch.tensor([0.5,0.5])
        assert kl_q_hat_pi(q,p).item() != pytest.approx(kl_q_hat_pi(p,q).item(), abs=1e-4)

    def test_scalar(self):
        assert kl_q_hat_pi(torch.tensor([0.4,0.6]), torch.tensor([0.5,0.5])).ndim == 0


# ---------------------------------------------------------------------------
# E) run_capra_forward_pass stub tests
# ---------------------------------------------------------------------------

class TestForwardPassStub:
    """Monkeypatch _run_anchor_forward to avoid GPU/model dependencies."""

    def _patch(self, monkeypatch, anchor_loss: float = 0.42):
        import vla_scripts.finetune_capra as fc
        monkeypatch.setattr(
            fc, "_run_anchor_forward",
            lambda **kw: (
                torch.tensor(anchor_loss),
                {"loss_value": anchor_loss, "curr_action_l1_loss": anchor_loss},
                torch.zeros(2, 8, 7),  # predicted_actions stub
            ),
        )

    # a) baseline mode -- CAPRA disabled
    def test_baseline_loss(self, monkeypatch):
        self._patch(monkeypatch, 0.42)
        loss, _ = run_capra_forward_pass(
            vla=None, action_head=None, noisy_action_projector=None,
            proprio_projector=None, batch=_fake_batch(), action_tokenizer=None,
            device_id=0, cfg=_baseline_cfg(),
            num_patches=256, capra_records=[], gradient_step_idx=0,
        )
        assert loss.item() == pytest.approx(0.42)

    def test_baseline_capra_loss_zero(self, monkeypatch):
        self._patch(monkeypatch)
        _, m = run_capra_forward_pass(
            vla=None, action_head=None, noisy_action_projector=None,
            proprio_projector=None, batch=_fake_batch(), action_tokenizer=None,
            device_id=0, cfg=_baseline_cfg(),
            num_patches=256, capra_records=[], gradient_step_idx=0,
        )
        assert m["capra_loss"] == pytest.approx(0.0)

    # b) CAPRA enabled but no records => anchor only
    def test_capra_no_records(self, monkeypatch):
        self._patch(monkeypatch, 0.55)
        loss, m = run_capra_forward_pass(
            vla=None, action_head=None, noisy_action_projector=None,
            proprio_projector=None, batch=_fake_batch(), action_tokenizer=None,
            device_id=0, cfg=_capra_cfg(),
            num_patches=256, capra_records=[], gradient_step_idx=100,
        )
        assert loss.item() == pytest.approx(0.55)
        assert m["capra_loss"] == pytest.approx(0.0)

    # c) warmup suppresses CAPRA even with records
    def test_warmup_suppresses_capra(self, monkeypatch):
        self._patch(monkeypatch, 0.3)
        loss, m = run_capra_forward_pass(
            vla=None, action_head=None, noisy_action_projector=None,
            proprio_projector=None, batch=_fake_batch(), action_tokenizer=None,
            device_id=0, cfg=_capra_cfg(capra_warmup_steps=9999),
            num_patches=256, capra_records=[_record()], gradient_step_idx=0,
        )
        assert loss.item() == pytest.approx(0.3)
        assert m["capra_loss"] == pytest.approx(0.0)

    # d) CAPRA active: records present, warmup=0, gamma=0 => uniform prior
    def test_capra_active_ratio_positive(self, monkeypatch):
        self._patch(monkeypatch, 0.1)
        _, m = run_capra_forward_pass(
            vla=None, action_head=None, noisy_action_projector=None,
            proprio_projector=None, batch=_fake_batch(), action_tokenizer=None,
            device_id=0, cfg=_capra_cfg(lam=1.0),
            num_patches=256, capra_records=[_record(w=1.0)] * 2, gradient_step_idx=0,
        )
        assert m["activation_ratio"] > 0.0

    def test_capra_active_loss_nonneg(self, monkeypatch):
        self._patch(monkeypatch, 0.1)
        loss, m = run_capra_forward_pass(
            vla=None, action_head=None, noisy_action_projector=None,
            proprio_projector=None, batch=_fake_batch(), action_tokenizer=None,
            device_id=0, cfg=_capra_cfg(lam=1.0),
            num_patches=256, capra_records=[_record(w=1.0)] * 2, gradient_step_idx=0,
        )
        assert loss.item() >= 0.0
        assert m["capra_loss"] >= 0.0

    # Required metric keys always present regardless of mode
    def test_required_keys_baseline(self, monkeypatch):
        self._patch(monkeypatch)
        _, m = run_capra_forward_pass(
            vla=None, action_head=None, noisy_action_projector=None,
            proprio_projector=None, batch=_fake_batch(), action_tokenizer=None,
            device_id=0, cfg=_baseline_cfg(),
            num_patches=256, capra_records=[], gradient_step_idx=0,
        )
        for k in ["anchor_loss", "capra_loss", "activation_ratio", "mean_w_t", "mean_delta_t"]:
            assert k in m, f"missing key: {k}"

    def test_required_keys_capra(self, monkeypatch):
        self._patch(monkeypatch)
        _, m = run_capra_forward_pass(
            vla=None, action_head=None, noisy_action_projector=None,
            proprio_projector=None, batch=_fake_batch(), action_tokenizer=None,
            device_id=0, cfg=_capra_cfg(),
            num_patches=256, capra_records=[_record()], gradient_step_idx=0,
        )
        for k in ["anchor_loss", "capra_loss", "activation_ratio", "mean_w_t", "mean_delta_t"]:
            assert k in m, f"missing key: {k}"


# ---------------------------------------------------------------------------
# F) CAPRADatasetReader
# ---------------------------------------------------------------------------

class TestReaderEmpty:
    def test_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = CAPRADatasetReader(tmp, "no_suite", FinetuneCAPRAConfig(cache_root=Path(tmp)))
            assert r.is_empty()

    def test_len_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = CAPRADatasetReader(tmp, "no_suite", FinetuneCAPRAConfig(cache_root=Path(tmp)))
            assert len(r) == 0

    def test_next_batch_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = CAPRADatasetReader(tmp, "no_suite", FinetuneCAPRAConfig(cache_root=Path(tmp)))
            assert r.next_batch(4) == []


class TestReaderNonEmpty:
    def _build_cache(self, cache_root: Path, n: int = 3):
        from experiments.robot.capra.mining.mining_cache import (
            CAPRAEpisodeCache, CAPRATimestepRecord, save_episode_cache,
        )
        cache = CAPRAEpisodeCache(
            episode_id="ep_test", task_description="pick up the mug",
            dataset_name="test_suite", total_steps=100,
        )
        rng = np.random.default_rng(42)
        for i in range(n):
            cache.append(CAPRATimestepRecord(
                step=i,
                candidate_actions=rng.standard_normal((4, 8, 7)).astype(np.float32),
                prior_weights=np.ones(4, np.float32) / 4,
                progress_values=np.array([0.8, 0.82, 0.6, 0.81], np.float32),
                footprint_values=np.array([0.3, 0.1, 0.5, 0.2], np.float32),
                equivalent_indices=np.array([0, 1], np.int32),
                p_max=0.82, delta_t=0.2, safest_action_idx=1,
                task_description="pick up the mug", episode_id="ep_test",
            ))
        save_episode_cache(cache, cache_root)

    def test_loads_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            cr = Path(tmp)
            self._build_cache(cr, 3)
            r = CAPRADatasetReader(tmp, "test_suite", FinetuneCAPRAConfig(cache_root=cr))
            assert not r.is_empty()
            assert len(r) == 3

    def test_next_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            cr = Path(tmp)
            self._build_cache(cr, 4)
            r = CAPRADatasetReader(tmp, "test_suite", FinetuneCAPRAConfig(cache_root=cr))
            assert len(r.next_batch(2)) == 2

    def test_wraps_around(self):
        with tempfile.TemporaryDirectory() as tmp:
            cr = Path(tmp)
            self._build_cache(cr, 2)
            r = CAPRADatasetReader(tmp, "test_suite", FinetuneCAPRAConfig(cache_root=cr))
            assert len(r.next_batch(5)) == 5

    def test_record_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            cr = Path(tmp)
            self._build_cache(cr, 2)
            r = CAPRADatasetReader(tmp, "test_suite", FinetuneCAPRAConfig(cache_root=cr))
            for rec in r.next_batch(2):
                assert "q_hat" in rec
                assert "weight" in rec
                assert "actions" in rec

