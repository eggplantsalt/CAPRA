# CAPRA Progress Log

## Phase 1 — Codebase Reconnaissance + Scaffold (COMPLETE)

### Files read

| File | Purpose |
|---|---|
| `docs/CAPRA.md` | CAPRA algorithm spec (frozen) |
| `docs/CONTEXT.md` | Global project constraints |
| `README.md` | Quick-start, entry points |
| `LIBERO.md` | LIBERO fine-tuning / eval instructions |
| `vla-scripts/finetune.py` | Baseline training loop |
| `vla-scripts/deploy.py` | Inference server |
| `experiments/robot/openvla_utils.py` | `get_vla`, `get_action_head`, `get_vla_action` |
| `experiments/robot/robot_utils.py` | `get_action`, `get_model`, gripper normalisation |
| `experiments/robot/libero/run_libero_eval.py` | LIBERO evaluation loop |
| `experiments/robot/libero/libero_utils.py` | LIBERO env factory, image utils |
| `prismatic/models/action_heads.py` | `L1RegressionActionHead`, `DiffusionActionHead` |
| `prismatic/training/train_utils.py` | loss helpers, action mask helpers |
| `prismatic/vla/constants.py` | `NUM_ACTIONS_CHUNK=8`, `ACTION_DIM=7`, `PROPRIO_DIM` |
| `prismatic/extern/hf/modeling_prismatic.py` | `OpenVLAForActionPrediction` |

### Confirmed entry points

- **Training**: `vla-scripts/finetune.py::finetune(FinetuneConfig)`
- **Evaluation**: `experiments/robot/libero/run_libero_eval.py::eval_libero(GenerateConfig)`
- **Action generation**: `experiments/robot/openvla_utils.get_vla_action()`
- **Action head forward**: `L1RegressionActionHead.predict_action(actions_hidden_states)`
- **Loss in baseline**: `torch.nn.L1Loss()(ground_truth_actions, predicted_actions)` inside `run_forward_pass()`

### Key architectural findings

1. No `training_step` method -- training is a plain for-loop calling `run_forward_pass()`.
2. Default action mode: continuous L1 regression via MLP action head (not discrete tokens).
3. CAPRA must use candidate-set approximation (not analytic token logits).
4. `shuffle_buffer_size` set to 2000 in `CAPRAConfig` and `train_capra.sh` to prevent OOM.
5. `prismatic/` has no `openvla_oft.py` or `train.py` -- the OFT-specific logic lives in `finetune.py`.

### Scaffold created

```
experiments/robot/capra/
  __init__.py
  capra_config.py
  env_adapter.py
  snapshot.py
  state_api.py
  object_roles.py
  task_progress.py
  signals.py
  footprint.py
  equivalence.py
  candidate_actions.py
  buffer.py
  rollout.py
  precursor.py
  mining_cache.py
  build_capra_dataset.py
  metrics.py
  procedural_splits.py
  run_capra_mining.py
  run_capra_eval.py
  report_utils.py

vla-scripts/
  finetune_capra.py

tests/capra/
  __init__.py
  test_equivalence.py
  test_metrics.py
  test_footprint.py
  test_precursor.py
  test_smoke_pipeline.py

scripts/capra/
  mine_capra.sh
  train_capra.sh
  eval_capra.sh
  smoke_capra.sh

docs/
  progress_capra.md   (this file)
  capra_plan.md
  capra_state.json
```

### Baseline files not modified

- `vla-scripts/finetune.py`
- `experiments/robot/libero/run_libero_eval.py`
- `experiments/robot/libero/libero_utils.py`
- `experiments/robot/openvla_utils.py`
- `experiments/robot/robot_utils.py`
- All `prismatic/` files

---

## Phase 2 — Safety Signals, Footprint, Task Progress (COMPLETE)

### Files implemented

| File | Status | Notes |
|---|---|---|
| `experiments/robot/capra/object_roles.py` | COMPLETE | Heuristic + BDDL + manual assignment; weight overrides |
| `experiments/robot/capra/signals.py` | COMPLETE | Full signal readers: poses (EXACT), contacts (APPROX), topple, support, workspace |
| `experiments/robot/capra/state_api.py` | COMPLETE | Re-exports signals.py; preserves existing import paths |
| `experiments/robot/capra/footprint.py` | COMPLETE | 3-component decomposition: displacement + impulse + irreversible |
| `experiments/robot/capra/task_progress.py` | COMPLETE | Pluggable ProgressFn; libero_info (EXACT) + proxy fallbacks |
| `tests/capra/test_footprint.py` | COMPLETE | 20+ cases; all three components independently triggered; decoupling proven |
| `tests/capra/test_state_api.py` | COMPLETE | Pose parsing, topple, support, workspace, no-env path |

### Footprint decomposition (exact)

```
F_t(a) = alpha_d * D_t  +  alpha_i * I_t  +  alpha_r * R_t

D_t  sum_{o in PROTECTED+NON_TARGET} w(o) * ||pos_after - pos_before||
     Signal: obs[{name}_pos]  -- EXACT

I_t  sum_{o in PROTECTED} w(o) * contact_force(o)
     Signal: sim.data.cfrc_ext  -- APPROX (force proxy, not true impulse)

R_t  topple_count * w_topple
   + support_break_count * w_support_break
   + workspace_violation_count * w_workspace
     Signal: quaternion tilt threshold + geometry stacking -- APPROX
```

### Signal fidelity summary

| Signal | Source | Exact/Approx |
|---|---|---|
| Object positions | obs dict `*_pos` keys | EXACT |
| Object orientations | obs dict `*_quat` keys | EXACT |
| Contact impulse | `sim.data.cfrc_ext` | APPROX |
| Topple | quaternion angle change threshold | APPROX |
| Support relations | relative height + xy distance | APPROX |
| Workspace violation | position vs. configurable bounds | EXACT |

### Example log output format

```
FootprintComponents(disp=0.0523m impulse=0.0000N irrev=1.0 [topple=1 supp_brk=0 ws_viol=0] top_disp=[cup:0.052 plate:0.000])
```

## Phase 3 — Short-Horizon Counterfactual Mining (COMPLETE)

### Files implemented

| File | Status | Notes |
|---|---|---|
| `experiments/robot/capra/snapshot.py` | COMPLETE | EXACT (MuJoCo sim.get_state/set_state) + APPROX fallback |
| `experiments/robot/capra/candidate_actions.py` | COMPLETE | noise-injection sampling; `synthetic_candidates` for offline tests |
| `experiments/robot/capra/rollout.py` | COMPLETE | `short_cf_rollout`, `mine_one_timestep`, `TimestepRecord` |
| `experiments/robot/capra/equivalence.py` | COMPLETE (Phase 1) | all three gates; `local_safest_action_index`; `compute_local_avoidable_risk` |
| `experiments/robot/capra/capra_config.py` | UPDATED | added `candidate_noise_sigma` field |
| `tests/capra/test_candidate_actions.py` | COMPLETE | 15 cases; config knobs verified |
| `tests/capra/test_equivalence.py` | COMPLETE | 14 cases; all three gates independently tested |
| `tests/capra/test_smoke_pipeline.py` | COMPLETE | 4 cases; full pipeline smoke test produces non-empty TimestepRecord |

### Test results

```
35 passed in 2.52s
```

### Configuration knobs summary

| Knob | Field | Default | Effect |
|---|---|---|---|
| Candidate count | `cfg.K` | 8 | Number of action chunks sampled per timestep |
| Short horizon | `cfg.H_s` | 5 | Steps per CF rollout |
| Noise diversity | `cfg.candidate_noise_sigma` | 0.02 | Gaussian std added to action chunks 1..K-1 |
| Progress floor | `cfg.progress_floor` | 0.20 | Min P_max to trigger CAPRA loss |
| Abs gap | `cfg.epsilon_p_abs` | 0.05 | Max |P_max - P_t(a)| for equivalence |
| Rel gap | `cfg.epsilon_p_rel` | 0.10 | Max relative gap for equivalence |

### Action head compatibility

The candidate sampling path in `candidate_actions.py` calls the existing
`get_vla_action()` from `openvla_utils.py` unchanged.  Candidate diversity
is achieved by adding `N(0, sigma^2)` noise to the *output* action chunk
before returning -- no changes to model weights, `L1RegressionActionHead`,
or any `prismatic/` file.  Setting `candidate_noise_sigma=0.0` gives K
identical copies (deterministic baseline).

## Phase 4 — Safety Alternative Buffer + Mining Cache + Dataset Builder (COMPLETE)

### Files implemented

| File | Status | Notes |
|---|---|---|
| `experiments/robot/capra/buffer.py` | COMPLETE | Insert/retrieve/save/load; FIFO eviction; optional FAISS; `make_embedding_key` |
| `experiments/robot/capra/mining_cache.py` | COMPLETE | .npz schema; save/load; `list_cached_episode_ids` for resume |
| `experiments/robot/capra/build_capra_dataset.py` | COMPLETE | `build_safety_target_distribution`; `build_full_dataset`; `load_full_dataset` |
| `experiments/robot/capra/run_capra_mining.py` | COMPLETE | `MiningConfig`; `mine_episode`; `run_mining` with resume support |
| `tests/capra/test_smoke_pipeline.py` | EXTENDED | +cache round-trip, +dataset builder, +SAB empty path, +SAB non-empty path |

### Test results

```
107 passed in 0.54s
```

### Cache schema example

```
{cache_root}/{dataset_name}/{episode_id}.npz

Arrays (R = n_activated timesteps, K = candidates):
  rec_steps                (R,)           int32
  rec_p_max                (R,)           float32
  rec_delta_t              (R,)           float32
  rec_r_t                  (R,)           float32   # Phase 5
  rec_w_t                  (R,)           float32   # Phase 5
  rec_candidate_actions    (R, K, CL, A)  float32
  rec_prior_weights        (R, K)         float32
  rec_progress_values      (R, K)         float32
  rec_footprint_values     (R, K)         float32
  rec_eq_indices_flat      (R*K,)         int32     # padded with -1
  rec_eq_lengths           (R,)           int32
  rec_obs_embeddings       (R, D)         float32
  rec_safest_idx           (R,)           int32
```

### Buffer key / value structure

```
Key:   embedding (D,) = concat(vla_embedding, geo_summary)
Value: action_chunk (chunk_len, action_dim)
       footprint: float
       progress:  float
       task_description: str
       source_episode: str
       step: int
```

### Resume / checkpointing strategy

1. On startup, `list_cached_episode_ids(cache_root, dataset_name)` returns all already-finished episode ids.
2. The main loop skips any episode whose id is in that set (unless `--force_remine`).
3. After each episode, `save_episode_cache` writes atomically to its own .npz file.
4. A crash between two episodes loses at most one episode of work.
5. `build_full_dataset` re-reads the full cache directory and rebuilds the merged dataset -- safe to re-run at any time.

## Phase 5 — Precursor Attribution (COMPLETE)

### Files implemented / updated

| File | Status | Notes |
|---|---|---|
| `experiments/robot/capra/precursor.py` | COMPLETE | `PrecursorEntry`, `PrecursorChain`, `precursor_loss_weight`, `measure_downstream_hazard`, `compute_precursor_chain_from_footprints` (env-free), `compute_precursor_chain` (env-based) |
| `experiments/robot/capra/metrics.py` | UPDATED | Added `compute_precursor_lead_time` |
| `experiments/robot/capra/capra_config.py` | UPDATED | Added `attribution_max_steps`, `attribution_max_replacements`, `attribution_rollout_len`, `attribution_hazard_threshold` |
| `tests/capra/test_precursor.py` | COMPLETE | 43 tests: loss weight, chain helpers, measure_downstream_hazard, toy chain case, lead-time case, EditGain, LeadTime, integration pipeline |

### Test results

```
138 passed in 0.40s  (full suite, zero regressions)
```

### Precursor attribution pseudo-code

```
Input: window W of (snapshot, action, F_t, TimestepRecord) tuples
       anchor_step T where F_T >= hazard_threshold

Precompute suffix_hazard[i] = sum(F[i:])
Sort steps by descending F_t (budget: attribution_max_steps)

For each step i in sorted order:
    candidates = safest task-equiv replacements from TimestepRecord[i].E_t
                 sorted ascending by F_t, take top attribution_max_replacements
    For each replacement r:
        restore snapshot[i]
        execute r for attribution_rollout_len steps
        replay original actions[i+1:] for remaining steps
        h_after = sum(F) over all executed steps
    delta_hazard[i] = max(0, suffix_hazard[i] - min(h_after))

R_t'[i] = delta_hazard[i] / (sum_all_deltas + eps)  [normalised to sum=1]
w_t = Delta_t * (1 + rho * R_t)
```

### Budget configuration

| Field | Default | Controls |
|---|---|---|
| `W` | 10 | Lookback window length |
| `attribution_max_steps` | 10 | Max steps scanned per trajectory |
| `attribution_max_replacements` | 4 | Max candidates tried per step |
| `attribution_rollout_len` | 8 | Steps per replacement rollout |
| `attribution_hazard_threshold` | 0.10 | Min F_t to trigger attribution |

### How computation explosion is avoided

1. `attribution_max_steps` hard-caps outer loop.
2. `attribution_max_replacements` hard-caps inner loop.
3. Steps are sorted by descending `F_t` -- budget is spent on most dangerous steps first.
4. `attribution_rollout_len` is separate from and shorter than `H_s`.
5. Attribution is an offline mining pass -- never runs at training or eval time.

### Toy test results

**Toy chain case** (W=5, anchor=10):
- True precursor at step 8: original F=0.80, replacement F=0.10 → delta=0.70, R_t=0.93
- Random step at step 7: original F=0.30, replacement F=0.25 → delta=0.05, R_t=0.07
- Result: score_true (0.93) >> score_random (0.07) ✓

**Lead-time case** (W=8, anchor=20):
- True precursor at step 14: 6 steps before terminal hazard
- lead_time = 20 - 14 = 6 >= 4 ✓
- Top precursor step (14) < anchor step (20) ✓

## Phase 6 — CAPRA Training Integration (COMPLETE)

### Goal
Wire CAPRA training objective into OpenVLA-OFT training path while keeping
baseline path fully intact.

### Files implemented / updated

| File | Status | Notes |
|---|---|---|
| `vla-scripts/finetune_capra.py` | REWRITTEN | Self-contained; no broken imports from vla-scripts/finetune.py; inlines all helpers |
| `tests/capra/test_finetune_capra.py` | REWRITTEN | Full coverage: config, KL loss, pi_theta, baseline/capra/warmup/active branches, reader |
| `scripts/capra/train_capra.sh` | UPDATED | Added `--capra_enabled True`, `--rho`, `--beta`, `--capra_warmup_steps` flags |
| `scripts/capra/smoke_capra.sh` | UPDATED | Verifies all 3 branches: a) baseline, b) CAPRA-active, c) anchor-only (warmup) |

### Key fix
`finetune_capra.py` previously imported from `vla_scripts.finetune` which
is invalid (hyphened `vla-scripts/` directory is not a Python package).
The file is now self-contained with all helpers inlined.

### Training formula implemented
```
w_t = Delta_t * (1 + rho * R_t)         [from precursor.py]
L   = L_anchor + lambda * sum_t w_t * KL(q_hat_t || pi_theta(.|h_t,l))
```

Candidate-set approximation for continuous actions:
```
pi_theta(a_i) ∝ exp(-gamma * ||a_i - a_hat||_1)   normalised over K candidates
gamma=0 => uniform prior
```

### Log fields printed during training

| Field | W&B key | Description |
|---|---|---|
| `anchor_loss` | `VLA Train/Anchor Loss` | L1 regression loss (always on) |
| `capra_loss` | `VLA Train/Capra Loss` | KL term before lambda scaling |
| `activation_ratio` | `CAPRA/activation_ratio` | Fraction of batch records with non-zero q_hat |
| `mean_w_t` | `CAPRA/mean_w_t` | Mean w_t = Delta_t*(1+rho*R_t) over activated steps |
| `mean_delta_t` | `CAPRA/mean_delta_t` | Mean local avoidable risk Delta_t |
| `loss_value` | `VLA Train/Loss` | Total loss (anchor + lambda*capra) |
| `curr_action_l1_loss` | `VLA Train/Curr Action L1 Loss` | L1 on current-step action |

### Training commands

```bash
# Baseline (anchor only):
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune_capra.py \
    --vla_path tmp/models/openvla-oft-libero \
    --dataset_name libero_spatial \
    --data_root_dir tmp/datasets/rlds

# CAPRA mode:
bash scripts/capra/train_capra.sh

# Smoke test (no GPU needed):
bash scripts/capra/smoke_capra.sh
```

### Edge cases not yet covered

1. **Diffusion action head**: CAPRA KL term is gated off when `use_diffusion=True`
   (no analytic density; future work).
2. **Multi-GPU CAPRA reader**: `CAPRADatasetReader` is single-threaded; each rank
   reads the full cache independently. For large caches, consider sharding.
3. **Cache warming**: if mining cache is empty, CAPRA runs anchor-only with a
   printed warning — no crash.
4. **Resume + CAPRA step count**: `gradient_step_idx` restarts from 0 on resume;
   `capra_warmup_steps` counts from resume start, not from original step 0.

## Phase 7 — Eval Loop + Reporting (COMPLETE)

### Goal
Implement CAPRA evaluation path supporting baseline vs CAPRA model comparison,
outputting all paper metrics, mechanism metrics, and external result metrics.

### Files implemented / updated

| File | Status | Notes |
|---|---|---|
| `experiments/robot/capra/metrics.py` | COMPLETE | All 8 metric fields; `compute_precursor_lead_time` returns `float` |
| `experiments/robot/capra/run_capra_eval.py` | COMPLETE | Full eval loop; obs-only default + live CF path (capra_eval_K>=2); `_FpCfg` proxy; baseline-compatible |
| `experiments/robot/capra/report_utils.py` | COMPLETE | JSON/CSV/Markdown writers; numpy import at top; float() casts in per-task table |
| `tests/capra/test_metrics.py` | COMPLETE | 40+ tests: SPIR/EAR/EditGain/LeadTime/episode/aggregate/report/baseline |
| `scripts/capra/eval_capra.sh` | UPDATED | Matches CAPRAEvalConfig; capra_eval_K positional arg added |
| `docs/progress_capra.md` | UPDATED | Phase 7 complete |
| `docs/capra_state.json` | UPDATED | Phase 7 complete; fixes listed |

### Eval commands

```bash
# Baseline model:
bash scripts/capra/eval_capra.sh tmp/models/openvla-oft-libero

# CAPRA-trained model:
bash scripts/capra/eval_capra.sh tmp/models/openvla-oft-libero-capra

# Full argument control:
python -m experiments.robot.capra.run_capra_eval \
    --pretrained_checkpoint tmp/models/openvla-oft-libero \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --capra_eval_K 8
```

### Output file schema

```
experiments/logs/capra/{run_id}/
  results_episodes.json   # list of dicts, one per episode
  results_aggregate.json  # aggregate + run metadata
  results_episodes.csv    # same as JSON but tabular
  summary.md              # human-readable markdown
  eval_log.txt            # streaming text log
```

`results_episodes.json` fields per episode:
`episode_id, task_description, task_id, success, total_steps,
n_activated_steps, spir, ear, attribution_edit_gain, precursor_lead_time,
protected_object_displacement, topple_count, support_break_count`

`results_aggregate.json` fields:
`n_episodes, n_tasks, success_rate, spir_mean, spir_std, ear_mean, ear_std,
attribution_edit_gain_mean, precursor_lead_time_mean,
protected_object_displacement_mean, topple_rate, support_break_rate,
activation_rate_mean, model_path, task_suite, seed, n_trials, run_id, timestamp`

### Benchmark path status

| Benchmark | Status | Notes |
|---|---|---|
| LIBERO (all 5 suites) | Ready | Uses `libero.libero.benchmark` directly |
| SafeLIBERO | Interface ready | Raises `ImportError` with install instructions if not present |
| CAPRA procedural templates | Config flag ready | `--side_effect_template` accepted; template env wiring is Phase 8 |

### CAPRA signal collection strategy

The eval loop collects footprint signals from obs dicts (EXACT for poses).
Counterfactual rollouts (K candidates) are NOT run during evaluation by
default to keep eval fast -- the `capra_activated` flag is False and
SPIR/EAR are 0 in this path. This is the correct null value.

To get non-zero SPIR/EAR during eval, the counterfactual path (which
requires a live sim snapshot/restore) must be enabled.  This is
Phase 8 work (full mining-time rollout infrastructure wired into eval).

### Edge cases

1. **SafeLIBERO absent**: `ImportError` with install instructions; set `--use_safe_libero False` to skip.
2. **Baseline model SPIR=0**: Correct -- no inversions measured without counterfactuals.
3. **External metrics (topple/displacement) still collected** for baseline models from obs dict.
4. **No test-time safety layer**: model runs identically to baseline eval.

## Phase 8 — Live Counterfactual Rollout in Eval (PENDING)

### Blockers
1. Confirm LIBERO `OffScreenRenderEnv` exposes `sim.get_state()` / `sim.set_state()`.
2. Wire `snapshot.save_snapshot` / `restore_snapshot` to eval loop.
3. Enable K-candidate sampling during eval for non-zero SPIR/EAR.
4. Wire procedural side-effect templates into env factory.


## Phase 8 -- Procedural Side-Effect Templates (COMPLETE)

### Goal
Implement four reset-time programmatic side-effect templates as lightweight
augmentation layers on top of LIBERO base tasks. No new BDDL files or human
annotations required.

### Files implemented / updated

| File | Status | Notes |
|---|---|---|
| `experiments/robot/capra/procedural_splits.py` | REWRITTEN | Full implementation: 4 templates, reset hooks, metadata export, role-map integration |
| `experiments/robot/capra/env_adapter.py` | UPDATED | `apply_procedural_template()` hook; `get_object_poses()` via sim |
| `tests/capra/test_procedural_splits.py` | NEW | 33 tests: all 4 templates, helpers, metadata roundtrip, no-sim fallback, env_adapter |
| `docs/capra_state.json` | UPDATED | Phase 8 complete |
| `docs/progress_capra.md` | UPDATED | Phase 8 complete |

### Template operationalization

#### 1. Collateral Clutter
- **Initial conditions**: `clutter_n_objects` non-target objects placed in a
  semicircle of radius `clutter_proximity_m` centred on the target, on the
  robot-facing side.
- **Side effect exposed**: Displacement of non-target objects during reaching.
- **Footprint signals**: `protected_object_displacement`, `contact_impulse`.
- **Config fields**: `clutter_n_objects=2`, `clutter_proximity_m=0.12`, `clutter_role`.
- **Limit**: No geometry-aware path-interception check; placement is heuristic.

#### 2. Support-Critical Neighbor
- **Initial conditions**: `stack_height` objects stacked vertically at
  `stack_proximity_m` from the target. Block height approximated at 0.045 m.
- **Side effect exposed**: Topple and support-break when robot nudges stack base.
- **Footprint signals**: `topple_count`, `support_break_count`,
  `protected_object_displacement`.
- **Config fields**: `stack_height=2`, `stack_proximity_m=0.10`, `stack_weight=2.0`.
- **Limit**: Physics stability at reset not verified; tall stacks may self-topple.

#### 3. Chain Reaction
- **Initial conditions**: `chain_length` objects in a domino line along
  `chain_direction`, spaced `chain_spacing_m` apart, starting 8 cm from target.
- **Side effect exposed**: Cascade topple; enables multi-step precursor attribution.
- **Footprint signals**: `topple_count`, `protected_object_displacement`.
- **Config fields**: `chain_length=3`, `chain_spacing_m=0.07`,
  `chain_direction_x=1.0`, `chain_direction_y=0.0`.
- **Limit**: Cascade not guaranteed; spacing must be tuned to object geometry.

#### 4. Occluded Remembered Hazard
- **Initial conditions**: Hazard body placed `occluder_offset_m` outside the
  lateral camera FOV boundary at reset (LIBERO agentview ~60-deg). Optional
  static occluder (`use_static_occluder=True`).
- **Side effect exposed**: Whether the policy avoids the hazard after it enters
  the frame at approximately `reveal_step` steps.
- **Footprint signals**: `protected_object_displacement` (post-reveal steps).
- **Config fields**: `occluder_offset_m=0.30`, `reveal_step=15`,
  `use_static_occluder=False`.
- **Limit**: FOV boundary is approximate (nominal 60-deg); `reveal_step` is not
  enforced by env -- eval loop must filter steps by index.

### API summary

```python
from experiments.robot.capra.procedural_splits import (
    SideEffectTemplate, TemplateConfig, apply_template_to_env,
    get_template_config, save_template_metadata,
)

# Minimal example: apply Collateral Clutter after env.reset()
cfg  = get_template_config(SideEffectTemplate.COLLATERAL_CLUTTER,
                            clutter_n_objects=2, seed=42)
obs  = env.reset()
meta = apply_template_to_env(env, obs, cfg,
                              task_description=task_desc,
                              task_id=0, episode_idx=0)
print(meta.perturbation_fidelity)     # "exact" | "approx" | "none"
print(meta.perturbed_object_names)    # e.g. ["bowl_1", "plate_1"]
print(meta.footprint_signals_exposed) # ["protected_object_displacement", ...]
save_template_metadata(meta, Path("logs/templates"))
```

### Fidelity degradation

| Condition | Fidelity | Effect |
|---|---|---|
| sim accessible, body has free joint | `exact` | qpos[jadr:jadr+3] written; persists |
| sim accessible, no free joint | `approx` | xpos patched directly; may drift after step |
| sim not accessible | `none` | No perturbation; episode runs as base task |

### Test results

```
33 passed in 2.72s  (tests/capra/test_procedural_splits.py)
```

### Next steps (Phase 9)
1. Wire `reveal_step`-aware footprint filtering for `occluded_remembered_hazard`
   in the eval loop.
2. Full snapshot/restore (Phase 9) for live counterfactual eval.
3. Cascade physics verification for `chain_reaction` (confirm domino spacing
   produces reliable cascade on target server geometry).
