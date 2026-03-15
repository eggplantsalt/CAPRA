#!/usr/bin/env bash
# smoke_capra.sh -- Quick smoke test: verify pure-Python CAPRA logic.
# Does NOT require a GPU, a model checkpoint, or a LIBERO install.
#
# Usage:
#   bash scripts/capra/smoke_capra.sh

set -euo pipefail

echo "[smoke_capra] Running pure-Python import and logic checks..."

python - <<'PYEOF'
import sys

# ---- CAPRAConfig ----
from experiments.robot.capra.capra_config import CAPRAConfig
cfg = CAPRAConfig()
print(f"  CAPRAConfig OK  (lam={cfg.lam}, beta={cfg.beta}, K={cfg.K})")

# ---- FinetuneCAPRAConfig ----
from vla_scripts.finetune_capra import FinetuneCAPRAConfig
bcfg = FinetuneCAPRAConfig()
ccfg = FinetuneCAPRAConfig(capra_enabled=True)
assert bcfg.capra_enabled is False,  "baseline: capra_enabled must be False"
assert ccfg.capra_enabled is True,   "capra:    capra_enabled must be True"
assert bcfg.shuffle_buffer_size == 2000, "shuffle_buffer_size must be 2000"
print(f"  FinetuneCAPRAConfig OK  (baseline={bcfg.capra_enabled}, capra={ccfg.capra_enabled})")

# ---- Equivalence filter (pure numpy) ----
import numpy as np
from experiments.robot.capra.equivalence import (
    build_task_equivalent_set, local_safest_action_index, compute_local_avoidable_risk,
)
progress = np.array([0.80, 0.82, 0.60, 0.81])
actions  = np.random.default_rng(0).standard_normal((4, 8, 7)).astype(np.float32)
eq_actions, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
print(f"  equivalence OK  (E_t size={len(eq_idx)}, P_max={p_max:.3f})")

# ---- Safety target distribution ----
from experiments.robot.capra.build_capra_dataset import build_safety_target_distribution
footprints = np.array([0.1, 0.05, 0.3, 0.08])
prior      = np.ones(4) / 4
q_hat      = build_safety_target_distribution(footprints, eq_idx, prior, beta=cfg.beta)
assert abs(q_hat.sum() - 1.0) < 1e-5 or q_hat.sum() == 0, f"q_hat sums to {q_hat.sum()}"
print(f"  q_hat OK        (sum={q_hat.sum():.6f})")

# ---- CAPRA KL loss (CPU tensors) ----
import torch
from experiments.robot.capra.capra_loss import compute_capra_kl_loss, compute_pi_theta
rng = np.random.default_rng(1)
rec = {
    "q_hat":    q_hat,
    "weight":   np.float32(0.2),
    "actions":  rng.standard_normal((4, 8, 7)).astype(np.float32),
    "delta_t":  np.float32(0.1),
}
loss_val, metrics = compute_capra_kl_loss(
    capra_records=[rec],
    predicted_actions=torch.zeros(4, 8, 7),
    device=torch.device("cpu"),
    gamma=0.0,
)
assert loss_val.item() >= 0.0
print(f"  KL loss OK      (loss={loss_val.item():.6f}, ratio={metrics['activation_ratio']:.2f})")

# ---- a) baseline branch of run_capra_forward_pass ----
from vla_scripts.finetune_capra import run_capra_forward_pass
import vla_scripts.finetune_capra as fc

_orig = fc._run_anchor_forward
def _stub(**kw):
    return torch.tensor(0.25), {"loss_value": 0.25, "curr_action_l1_loss": 0.25}, torch.zeros(2,8,7)
fc._run_anchor_forward = _stub

fake_batch = {
    "input_ids": torch.zeros(2,10,dtype=torch.long),
    "attention_mask": torch.ones(2,10),
    "pixel_values": torch.zeros(2,3,224,224),
    "labels": torch.zeros(2,10,dtype=torch.long),
    "actions": torch.zeros(2,8,7),
}
loss_b, m_b = run_capra_forward_pass(
    vla=None, action_head=None, noisy_action_projector=None,
    proprio_projector=None, batch=fake_batch, action_tokenizer=None,
    device_id=0, cfg=FinetuneCAPRAConfig(capra_enabled=False,
        use_l1_regression=True, use_diffusion=False,
        use_proprio=False, use_film=False),
    num_patches=256, capra_records=[], gradient_step_idx=0,
)
assert abs(loss_b.item() - 0.25) < 1e-5
assert m_b["capra_loss"] == 0.0
print(f"  a) baseline branch OK  (loss={loss_b.item():.3f}, capra_loss={m_b['capra_loss']:.3f})")

# ---- b) CAPRA branch (warmup=0, records present) ----
loss_c, m_c = run_capra_forward_pass(
    vla=None, action_head=None, noisy_action_projector=None,
    proprio_projector=None, batch=fake_batch, action_tokenizer=None,
    device_id=0, cfg=FinetuneCAPRAConfig(capra_enabled=True,
        use_l1_regression=True, use_diffusion=False,
        use_proprio=False, use_film=False,
        lam=0.1, capra_warmup_steps=0, capra_gamma=0.0),
    num_patches=256, capra_records=[rec], gradient_step_idx=0,
)
assert loss_c.item() >= 0.0
assert m_c["activation_ratio"] > 0.0
print(f"  b) CAPRA branch OK     (loss={loss_c.item():.4f}, ratio={m_c['activation_ratio']:.2f})")

# ---- c) anchor-only: warmup suppresses CAPRA ----
loss_w, m_w = run_capra_forward_pass(
    vla=None, action_head=None, noisy_action_projector=None,
    proprio_projector=None, batch=fake_batch, action_tokenizer=None,
    device_id=0, cfg=FinetuneCAPRAConfig(capra_enabled=True,
        use_l1_regression=True, use_diffusion=False,
        use_proprio=False, use_film=False,
        lam=0.1, capra_warmup_steps=9999, capra_gamma=0.0),
    num_patches=256, capra_records=[rec], gradient_step_idx=0,
)
assert m_w["capra_loss"] == 0.0
print(f"  c) anchor-only (warmup) OK  (capra_loss={m_w['capra_loss']:.3f})")

fc._run_anchor_forward = _orig  # restore

# ---- SPIR / EAR ----
from experiments.robot.capra.metrics import compute_spir, compute_ear
chosen_f  = np.array([0.2, 0.1, 0.3])
min_f     = np.array([0.1, 0.1, 0.1])
activated = np.array([True, True, True])
spir = compute_spir(chosen_f, min_f, activated)
ear  = compute_ear(chosen_f - min_f, activated)
print(f"  SPIR={spir:.3f}  EAR={ear:.4f}")

# ---- Precursor weight ----
from experiments.robot.capra.precursor import precursor_loss_weight
w = precursor_loss_weight(delta_t=0.15, r_t=0.8, rho=cfg.rho)
print(f"  precursor_weight={w:.4f}")

print("\n[smoke_capra] All checks passed. baseline + CAPRA + anchor-only branches verified.")
PYEOF

echo "[smoke_capra] Done."
