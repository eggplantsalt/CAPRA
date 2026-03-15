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

## Phase 2 — Counterfactual Rollout Infrastructure (PENDING)

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
