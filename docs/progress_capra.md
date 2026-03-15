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

## Phase 5 — Precursor Attribution (PENDING)

### Blockers to resolve first

1. Confirm LIBERO `OffScreenRenderEnv` exposes `sim.get_state()` / `sim.set_state()` for exact snapshot.
2. Confirm contact/impulse API availability in MuJoCo version bundled with LIBERO.
3. Confirm BDDL predicate checker API for task progress signal.

### Next actions

1. Implement `snapshot.save_snapshot` / `restore_snapshot` against LIBERO MuJoCo backend.
2. Implement `state_api.read_state_signals` (poses, contacts, topple detection).
3. Implement `rollout.short_cf_rollout` (H_s step loop).
4. Implement `task_progress.libero_stage_progress`.
5. Implement `signals.aggregate_footprint_components`.
6. Wire everything into `run_capra_mining.py` mining loop.
