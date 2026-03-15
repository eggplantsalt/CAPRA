#!/usr/bin/env bash
# smoke_capra.sh
# Quick smoke test: verify imports and pure-Python logic run without errors.
# Does NOT require a GPU, a model checkpoint, or a LIBERO install.
#
# Usage:
#   bash scripts/capra/smoke_capra.sh

set -euo pipefail

echo "[smoke_capra] Running pure-Python import and logic checks..."

python - <<'EOF'
import sys

# Config
from experiments.robot.capra.capra_config import CAPRAConfig
cfg = CAPRAConfig()
print(f"  CAPRAConfig OK  (lam={cfg.lam}, beta={cfg.beta}, K={cfg.K})")

# Equivalence filter (pure numpy)
import numpy as np
from experiments.robot.capra.equivalence import (
    build_task_equivalent_set,
    local_safest_action_index,
    compute_local_avoidable_risk,
)
progress = np.array([0.80, 0.82, 0.60, 0.81])
actions  = np.random.randn(4, 8, 7).astype(np.float32)
eq_actions, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
print(f"  equivalence OK  (E_t size={len(eq_idx)}, P_max={p_max:.3f})")

# Safety target distribution
from experiments.robot.capra.build_capra_dataset import build_safety_target_distribution
footprints  = np.array([0.1, 0.05, 0.3, 0.08])
prior       = np.ones(4) / 4
q_hat       = build_safety_target_distribution(footprints, eq_idx, prior, beta=cfg.beta)
assert abs(q_hat.sum() - 1.0) < 1e-5 or q_hat.sum() == 0, f"q_hat sums to {q_hat.sum()}"
print(f"  q_hat OK        (sum={q_hat.sum():.6f})")

# SPIR / EAR
from experiments.robot.capra.metrics import compute_spir, compute_ear, aggregate_episode_metrics, EpisodeMetrics
chosen_f   = np.array([0.2, 0.1, 0.3])
min_f      = np.array([0.1, 0.1, 0.1])
activated  = np.array([True, True, True])
spir = compute_spir(chosen_f, min_f, activated)
ear  = compute_ear(chosen_f - min_f, activated)
print(f"  SPIR={spir:.3f}  EAR={ear:.4f}")

# Precursor weight
from experiments.robot.capra.precursor import precursor_loss_weight
w = precursor_loss_weight(delta_t=0.15, r_t=0.8, rho=cfg.rho)
print(f"  precursor_weight={w:.4f}")

# Object roles
from experiments.robot.capra.object_roles import ObjectRole, ObjectRoleMap
role_map = ObjectRoleMap(target=["bowl"], protected=["cup"], non_target=["plate"])
assert role_map.get_role("bowl") == ObjectRole.TARGET
assert role_map.get_role("cup")  == ObjectRole.PROTECTED
print(f"  ObjectRoleMap OK")

# Procedural templates
from experiments.robot.capra.procedural_splits import list_all_templates, get_template_config
templates = list_all_templates()
for t in templates:
    get_template_config(t)
print(f"  procedural_splits OK  ({len(templates)} templates)")

print("\n[smoke_capra] All pure-Python checks passed.")
EOF

echo "[smoke_capra] Done."
