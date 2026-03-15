"""Pure-Python CAPRA logic smoke test -- called by smoke_capra.sh."""
from __future__ import annotations
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import torch

# 1. CAPRAConfig
from experiments.robot.capra.core.capra_config import CAPRAConfig
cfg = CAPRAConfig()
print(f"  [1] CAPRAConfig OK  lam={cfg.lam} beta={cfg.beta} K={cfg.K}")

# 2. FinetuneCAPRAConfig
from vla_scripts.finetune_capra import FinetuneCAPRAConfig
bcfg = FinetuneCAPRAConfig()
ccfg = FinetuneCAPRAConfig(capra_enabled=True)
assert not bcfg.capra_enabled and ccfg.capra_enabled
assert bcfg.shuffle_buffer_size == 2000
print(f"  [2] FinetuneCAPRAConfig OK")

# 3. Equivalence filter
from experiments.robot.capra.core.equivalence import (
    build_task_equivalent_set, local_safest_action_index, compute_local_avoidable_risk,
)
progress = np.array([0.80, 0.82, 0.60, 0.81])
actions  = np.random.default_rng(0).standard_normal((4, 8, 7)).astype(np.float32)
eq_actions, eq_idx, p_max = build_task_equivalent_set(actions, progress, cfg)
print(f"  [3] equivalence OK  E_t={len(eq_idx)} P_max={p_max:.3f}")

# 4. Safety target distribution
from experiments.robot.capra.scene.build_capra_dataset import build_safety_target_distribution
footprints = np.array([0.1, 0.05, 0.3, 0.08])
prior      = np.ones(4) / 4
q_hat = build_safety_target_distribution(footprints, eq_idx, prior, beta=cfg.beta)
assert abs(q_hat.sum() - 1.0) < 1e-5 or q_hat.sum() == 0
print(f"  [4] q_hat OK  sum={q_hat.sum():.6f}")

# 5. CAPRA KL loss
from experiments.robot.capra.core.capra_loss import compute_capra_kl_loss
rng = np.random.default_rng(1)
rec = {
    "q_hat":   q_hat,
    "weight":  np.float32(0.2),
    "actions": rng.standard_normal((4, 8, 7)).astype(np.float32),
    "delta_t": np.float32(0.1),
}
loss_val, met = compute_capra_kl_loss(
    capra_records=[rec], predicted_actions=torch.zeros(4, 8, 7),
    device=torch.device("cpu"), gamma=0.0,
)
assert loss_val.item() >= 0.0
print(f"  [5] KL loss OK  loss={loss_val.item():.6f} ratio={met['activation_ratio']:.2f}")

# 6. Training branches
from vla_scripts.finetune_capra import run_capra_forward_pass
import vla_scripts.finetune_capra as fc
_orig = fc._run_anchor_forward
def _stub(**kw):
    return torch.tensor(0.25), {"loss_value": 0.25, "curr_action_l1_loss": 0.25}, torch.zeros(2,8,7)
fc._run_anchor_forward = _stub
fake = {
    "input_ids": torch.zeros(2,10,dtype=torch.long),
    "attention_mask": torch.ones(2,10),
    "pixel_values": torch.zeros(2,3,224,224),
    "labels": torch.zeros(2,10,dtype=torch.long),
    "actions": torch.zeros(2,8,7),
}
base_kw = dict(vla=None, action_head=None, noisy_action_projector=None,
               proprio_projector=None, batch=fake, action_tokenizer=None,
               device_id=0, num_patches=256, gradient_step_idx=0)
# 6a baseline
loss_b, m_b = run_capra_forward_pass(**base_kw,
    cfg=FinetuneCAPRAConfig(capra_enabled=False, use_l1_regression=True,
        use_diffusion=False, use_proprio=False, use_film=False),
    capra_records=[])
assert abs(loss_b.item()-0.25)<1e-5 and m_b["capra_loss"]==0.0
print(f"  [6a] baseline OK  capra_loss={m_b['capra_loss']:.3f}")
# 6b CAPRA
loss_c, m_c = run_capra_forward_pass(**base_kw,
    cfg=FinetuneCAPRAConfig(capra_enabled=True, use_l1_regression=True,
        use_diffusion=False, use_proprio=False, use_film=False,
        lam=0.1, capra_warmup_steps=0, capra_gamma=0.0),
    capra_records=[rec])
assert loss_c.item()>=0.0 and m_c["activation_ratio"]>0.0
print(f"  [6b] CAPRA OK  ratio={m_c['activation_ratio']:.2f}")
# 6c warmup
loss_w, m_w = run_capra_forward_pass(**base_kw,
    cfg=FinetuneCAPRAConfig(capra_enabled=True, use_l1_regression=True,
        use_diffusion=False, use_proprio=False, use_film=False,
        lam=0.1, capra_warmup_steps=9999, capra_gamma=0.0),
    capra_records=[rec])
assert m_w["capra_loss"]==0.0
print(f"  [6c] warmup OK  capra_loss={m_w['capra_loss']:.3f}")
fc._run_anchor_forward = _orig

# 7. SPIR / EAR
from experiments.robot.capra.eval.metrics import compute_spir, compute_ear
chosen_f  = np.array([0.2, 0.1, 0.3])
min_f     = np.array([0.1, 0.1, 0.1])
activated = np.array([True, True, True])
spir = compute_spir(chosen_f, min_f, activated)
ear  = compute_ear(chosen_f - min_f, activated)
print(f"  [7] SPIR={spir:.3f}  EAR={ear:.4f}")

# 8. EpisodeMetrics + AggregateMetrics + report writers
from experiments.robot.capra.eval.metrics import (
    TimestepEvalRecord, compute_episode_metrics, aggregate_episode_metrics
)
from experiments.robot.capra.eval.report_utils import save_all_reports
records = [TimestepEvalRecord(step=i, chosen_footprint=0.2,
    min_equivalent_footprint=0.1, capra_activated=True, delta_t=0.1)
    for i in range(5)]
ep  = compute_episode_metrics(records, episode_id="ep0", task_id=0, success=True)
agg = aggregate_episode_metrics([ep, ep], n_tasks=1)
assert agg.n_episodes==2 and agg.success_rate==1.0
with tempfile.TemporaryDirectory() as td:
    save_all_reports(agg, [ep,ep], pathlib.Path(td),
                     model_path="tmp/model", task_suite="libero_spatial")
    n_files = len(list(pathlib.Path(td).iterdir()))
    assert n_files == 4, f"expected 4 files got {n_files}"
print(f"  [8] EpisodeMetrics+reports OK  spir={ep.spir:.3f}  files={n_files}")

# 9. Precursor weight
from experiments.robot.capra.scene.precursor import precursor_loss_weight
w = precursor_loss_weight(delta_t=0.15, r_t=0.8, rho=cfg.rho)
print(f"  [9] precursor_weight={w:.4f}")

# 10. Procedural splits (mock env)
from experiments.robot.capra.eval.procedural_splits import (
    SideEffectTemplate, get_template_config, apply_template_to_env,
    list_all_templates,
)

class _MM:
    def __init__(self, bods):
        self._n=[b[0] for b in bods]; self.nbody=len(bods)
        self.body_jntadr=[i if b[1] else -1 for i,b in enumerate(bods)]
        self.jnt_type=[0]*len(bods)
    def body_name2id(self,n): return self._n.index(n)
    def body_id2name(self,i): return self._n[i]
class _MSim:
    def __init__(self):
        bods=[("mug_1",True),("bowl_1",True),("plate_1",True),("box_1",True),("robot0",False)]
        self.model=_MM(bods)
        self.data=type("D",(),{"qpos":np.zeros(35),"body_xpos":np.zeros((5,3))})()
    def forward(self): pass
class _MEnv:
    def __init__(self): self.sim=_MSim()

obs={}
for i,n in enumerate(["mug_1","bowl_1","plate_1","box_1"]):
    obs[f"{n}_pos"]=np.array([0.1*i,0.0,0.05])
    obs[f"{n}_quat"]=np.array([0.0,0.0,0.0,1.0])
tdesc="pick the mug_1 and place it"

for t in list_all_templates():
    meta=apply_template_to_env(_MEnv(),obs,get_template_config(t,seed=0),task_description=tdesc)
    assert meta.template==t.value
    print(f"  [10] {t.value}: fidelity={meta.perturbation_fidelity} n={len(meta.perturbed_object_names)}")

# 11. CAPRAEnvAdapter no-sim path
from experiments.robot.capra.mining.env_adapter import CAPRAEnvAdapter, EnvConfig
class _FR: pass
adapter=CAPRAEnvAdapter(_FR(), EnvConfig(side_effect_template="collateral_clutter"))
meta2=adapter.apply_procedural_template(obs, task_description=tdesc)
assert meta2 is not None and meta2.perturbation_fidelity=="none"
print(f"  [11] CAPRAEnvAdapter no-sim OK  fidelity={meta2.perturbation_fidelity}")

print("\n[smoke_capra] All checks passed. baseline+CAPRA+templates verified.")
