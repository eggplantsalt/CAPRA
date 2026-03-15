# CAPRA: Counterfactual Action Preference with Risk Avoidance

> Built on top of [OpenVLA-OFT](https://github.com/openvla/openvla-oft).  
> CAPRA adds a safety-preference training signal that penalises actions with
> unnecessarily high environmental footprint when a task-equivalent safer
> alternative exists.

---

## Table of Contents

1. [Project goal](#1-project-goal)
2. [Directory layout](#2-directory-layout)
3. [Quick-start: minimal vertical slice](#3-quick-start-minimal-vertical-slice)
4. [Step-by-step commands](#4-step-by-step-commands)
5. [Smoke test](#5-smoke-test)
6. [Configuration reference](#6-configuration-reference)
7. [Output layout](#7-output-layout)
8. [Known limitations and blockers](#8-known-limitations-and-blockers)
9. [Handover checklist](#9-handover-checklist)

---

## 1. Project goal

Train a VLA (OpenVLA-OFT) that prefers low-footprint actions when multiple
task-equivalent choices exist.  CAPRA does **not** add a test-time safety
filter -- it changes what the model learns to prefer.

Key metrics produced at eval time:

| Metric | Meaning |
|---|---|
| SPIR | Fraction of steps where model chose higher-footprint action than safest equivalent |
| EAR / J\_AR | Expected avoidable footprint per activated step |
| EditGain | Footprint reduction from applying top precursor replacement |
| LeadTime | Steps between first warning signal and terminal hazard |
| SR | Task success rate (unchanged from baseline) |

---

## 2. Directory layout

```
openvla-oft/
  docs/
    CAPRA.md              Algorithm spec (frozen reference)
    CONTEXT.md            Global engineering constraints
    README_CAPRA.md       This file
    progress_capra.md     Phase-by-phase implementation log
    capra_state.json      Machine-readable phase state
    capra_plan.md         Original planning document

  experiments/robot/capra/    All CAPRA library code (organised into subpackages)
    core/                     核心算法
      capra_config.py         CAPRAConfig -- all hyperparameters
      capra_loss.py           CAPRA KL loss
      equivalence.py          Task-equivalence set E_t
      footprint.py            3-component footprint F_t
      signals.py              StateSignals (poses, contacts, topple)
      state_api.py            Re-export shim for signals
    scene/                    场景语义
      object_roles.py         Object role taxonomy (TARGET/PROTECTED/NON_TARGET)
      task_progress.py        Task progress estimators
      precursor.py            PrecursorChain and attribution
      build_capra_dataset.py  Safety target distribution builder
    mining/                   离线挖掘
      run_capra_mining.py     Mining entry point (CLI)
      mining_cache.py         Disk-backed supervision cache
      buffer.py               Training sample reader
      rollout.py              Short-horizon CF rollout
      candidate_actions.py    Candidate action samplers
      snapshot.py             MuJoCo snapshot/restore
      env_adapter.py          CAPRAEnvAdapter + procedural template hook
    eval/                     评估与报告
      run_capra_eval.py       Eval entry point (CLI)
      metrics.py              SPIR, EAR, EpisodeMetrics, AggregateMetrics
      report_utils.py         JSON / CSV / Markdown writers
      procedural_splits.py    4 reset-time side-effect templates
    *.py (shims)              Backward-compatibility re-export shims

  vla-scripts/
    finetune.py               Baseline training (unchanged)
    finetune_capra.py         CAPRA training entry point

  scripts/capra/
    mine_capra.sh             Offline mining
    train_capra.sh            CAPRA fine-tuning
    eval_capra.sh             CAPRA evaluation
    smoke_capra.sh            Pure-Python logic smoke test

  tests/capra/
    test_*.py                 Unit tests (pure Python, no GPU)

  tmp/                        (gitignored) local artefacts
    models/                   Checkpoints
    datasets/rlds/            RLDS datasets
    capra_cache/              Mined supervision cache

  runs/                       Training run outputs
  experiments/logs/capra/     Eval run outputs
```

---

## 3. Quick-start: minimal vertical slice

```bash
# 0. Install (on server)
conda activate openvla
pip install -e .

# 1. Mine supervision (50 episodes)
bash scripts/capra/mine_capra.sh \
    tmp/models/openvla-oft-libero libero_spatial tmp/capra_cache

# 2. Train CAPRA model
bash scripts/capra/train_capra.sh \
    tmp/models/openvla-oft-libero libero_spatial tmp/capra_cache

# 3. Evaluate CAPRA model
bash scripts/capra/eval_capra.sh \
    runs/CAPRA-.../checkpoints/latest libero_spatial 50

# 3b. Evaluate baseline (obs-only CAPRA signals, no counterfactuals)
bash scripts/capra/eval_capra.sh \
    tmp/models/openvla-oft-libero libero_spatial 50
```

---

## 4. Step-by-step commands

### 4.1 Mining

```bash
bash scripts/capra/mine_capra.sh [CHECKPOINT] [DATASET] [CACHE_ROOT]

# Full CLI:
python -m experiments.robot.capra.run_capra_mining \
    --pretrained_checkpoint tmp/models/openvla-oft-libero \
    --dataset_name          libero_spatial \
    --cache_root            tmp/capra_cache \
    --num_mining_episodes   50 \
    --K 8 --H_s 5 --W 10 --seed 7
```

Outputs to `tmp/capra_cache/libero_spatial/` -- one JSON per episode.

### 4.2 Training

```bash
bash scripts/capra/train_capra.sh [CHECKPOINT] [DATASET] [CACHE_ROOT]

# CAPRA mode (full CLI):
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune_capra.py \
    --capra_enabled True --lam 0.1 --rho 0.5 --beta 1.0 \
    --capra_warmup_steps 500 --shuffle_buffer_size 2000 \
    --vla_path tmp/models/openvla-oft-libero \
    --dataset_name libero_spatial --cache_root tmp/capra_cache

# Baseline mode (anchor loss only):
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune_capra.py \
    --capra_enabled False \
    --vla_path tmp/models/openvla-oft-libero \
    --dataset_name libero_spatial
```

**Critical**: always `--shuffle_buffer_size 2000`.  
Higher values will exhaust container memory via TF RLDS loader.

Outputs to `runs/CAPRA-{dataset}-{datetime}/`.

### 4.3 Evaluation

```bash
# Positional: CHECKPOINT SUITE TRIALS CF_K
bash scripts/capra/eval_capra.sh tmp/models/openvla-oft-libero libero_spatial 50 0

# With live counterfactual eval (requires MuJoCo sim.get_state):
bash scripts/capra/eval_capra.sh tmp/models/openvla-oft-libero libero_spatial 50 8

# Full CLI:
python -m experiments.robot.capra.run_capra_eval \
    --pretrained_checkpoint tmp/models/openvla-oft-libero \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 --capra_eval_K 0
```

Default `capra_eval_K=0`: obs-only path, SPIR=EAR=0 (correct null value --
no comparison set without counterfactuals).  
Set `capra_eval_K>=2` for non-zero SPIR/EAR (snapshot/restore required).

Outputs to `experiments/logs/capra/CAPRA-EVAL-{suite}-{datetime}/`.

### 4.4 Procedural side-effect splits

Four templates available: `collateral_clutter`, `support_critical_neighbor`,
`chain_reaction`, `occluded_remembered_hazard`.

```bash
# Eval with a procedural template applied at each reset:
bash scripts/capra/eval_capra.sh \
    tmp/models/openvla-oft-libero libero_spatial 50 0 collateral_clutter

# Python API:
from experiments.robot.capra.procedural_splits import (
    SideEffectTemplate, get_template_config, apply_template_to_env,
    save_template_metadata,
)
cfg  = get_template_config(SideEffectTemplate.CHAIN_REACTION, chain_length=3)
obs  = env.reset()
meta = apply_template_to_env(env, obs, cfg, task_description=task_desc)
print(meta.perturbation_fidelity)      # exact / approx / none
print(meta.footprint_signals_exposed)  # ["topple_count", ...]
save_template_metadata(meta, Path("logs/templates"))
```

Template fidelity:

| Condition | Fidelity |
|---|---|
| MuJoCo free-joint qpos writable | `exact` |
| sim accessible, no free joint | `approx` |
| sim not accessible | `none` (episode runs as base task) |


---

## 5. Smoke test

```bash
# Pure-Python check (no GPU, no LIBERO needed):
bash scripts/capra/smoke_capra.sh
```

Verifies: CAPRAConfig, FinetuneCAPRAConfig, equivalence filter, safety target
distribution, KL loss, baseline/CAPRA/anchor-only training branches, SPIR/EAR
metrics, procedural template (all 4 templates, mock env), and report writers.

---

## 6. Configuration reference

### CAPRAConfig (`experiments/robot/capra/capra_config.py`)

| Field | Default | Meaning |
|---|---|---|
| `K` | 8 | Candidate action chunks per step |
| `H_s` | 5 | Short CF rollout horizon |
| `W` | 10 | Precursor attribution lookback window |
| `lam` | 0.1 | CAPRA loss weight vs anchor loss |
| `rho` | 0.5 | Precursor attribution upweight factor |
| `beta` | 1.0 | Temperature for safety target distribution |
| `alpha_d/i/r` | 1/1/2 | Footprint component weights |
| `epsilon_p_abs` | 0.05 | Absolute task-progress gap threshold |
| `epsilon_p_rel` | 0.10 | Relative task-progress gap threshold |
| `progress_floor` | 0.20 | Min P_max to activate CAPRA loss |
| `shuffle_buffer_size` | 2000 | **Keep at 2000** -- higher causes OOM |

### FinetuneCAPRAConfig (`vla-scripts/finetune_capra.py`)

Extends `CAPRAConfig` with all training fields (`batch_size`, `learning_rate`,
`max_steps`, `use_lora`, `lora_rank`, etc.).

### CAPRAEvalConfig (`experiments/robot/capra/run_capra_eval.py`)

| Field | Default | Meaning |
|---|---|---|
| `capra_eval_K` | 0 | 0=obs-only; >=2=live CF eval |
| `capra_eval_sigma` | 0.02 | Noise std for CF candidate sampling |
| `side_effect_template` | None | Procedural template name |
| `task_suite_name` | libero_spatial | LIBERO suite |
| `num_trials_per_task` | 50 | Episodes per task |

---

## 7. Output layout

```
experiments/logs/capra/
  CAPRA-EVAL-{suite}-{datetime}/
    results_episodes.json    per-episode metrics
    results_aggregate.json   aggregate + std + metadata
    results_episodes.csv     tabular
    summary.md               markdown with per-task breakdown
    eval_log.txt             streaming log

tmp/capra_cache/{dataset}/
  episode_{id}.json          mined supervision records

runs/CAPRA-{dataset}-{datetime}/
  checkpoints/               model weights
  metrics/                   training loss curves
```

---

## 8. Known limitations and blockers

### Blockers (require server validation)

1. **snapshot/restore**: `run_capra_eval.py` CF path calls
   `save_snapshot` / `restore_snapshot` from `snapshot.py`.  These require
   `env.sim.get_state()` / `set_state()` (mujoco-py API).  Must verify
   LIBERO `OffScreenRenderEnv` exposes this on the target server.
   *Workaround*: `capra_eval_K=0` (obs-only, always works).

2. **SafeLIBERO**: `--use_safe_libero True` requires `pip install safe-libero`.
   Not installed by default.  Set `--use_safe_libero False` to use plain LIBERO.

3. **RLDS dataset**: Datasets must be downloaded from HuggingFace before training.
   `huggingface-cli download openvla/libero-spatial-no-noops-rlds --local-dir tmp/datasets/rlds/libero_spatial`

4. **Model checkpoint**: Must download or provide an OFT checkpoint at
   `tmp/models/openvla-oft-libero` before mining or eval.

### Known approximations (do not block running)

5. **Footprint signals**: Contact impulse uses obs-dict `robot0_contact_force`
   if available; otherwise falls back to zero (APPROX).  Topple detection uses
   Z-axis orientation threshold, not physics engine contact events.

6. **Procedural template fidelity**: `occluded_remembered_hazard` FOV boundary
   is approximate (nominal 60-deg agentview).  `chain_reaction` cascade is not
   physics-verified -- spacing may need tuning.

7. **SPIR/EAR=0 in default eval**: This is correct -- no comparison set without
   counterfactuals.  Use `capra_eval_K>=2` for non-zero values.

8. **Task progress proxy**: CF eval uses `done` flag as progress proxy (1.0 if
   done, 0.5 otherwise).  A proper multi-step progress estimate (Phase 9) would
   improve equivalence-set quality.

---

## 9. Handover checklist

### Before handing to new engineer

- [ ] Server has conda env `openvla` with all deps installed (`pip install -e .`)
- [ ] `tmp/models/openvla-oft-libero` exists (download from HuggingFace)
- [ ] `tmp/datasets/rlds/libero_spatial` exists
- [ ] `bash scripts/capra/smoke_capra.sh` passes (pure-Python, no GPU)
- [ ] `bash scripts/capra/mine_capra.sh` runs without error
- [ ] `bash scripts/capra/train_capra.sh` starts training loop
- [ ] `bash scripts/capra/eval_capra.sh` produces output in `experiments/logs/capra/`

### Key files to understand first

1. `docs/CAPRA.md` -- algorithm spec (start here)
2. `experiments/robot/capra/capra_config.py` -- all knobs
3. `vla-scripts/finetune_capra.py` -- training entry point
4. `experiments/robot/capra/run_capra_eval.py` -- eval entry point
5. `experiments/robot/capra/metrics.py` -- metric definitions

### How baseline and CAPRA differ

| Aspect | Baseline (`finetune.py`) | CAPRA (`finetune_capra.py --capra_enabled True`) |
|---|---|---|
| Loss | L1 regression (anchor) | anchor + weighted KL divergence |
| Data | RLDS dataset | RLDS + mined supervision cache |
| Test time | identical | identical (no safety filter) |
| Eval | `run_libero_eval.py` | `run_capra_eval.py` (adds SPIR/EAR) |

