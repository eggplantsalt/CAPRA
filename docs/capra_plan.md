# CAPRA Implementation Plan

## 1. Real entry points (confirmed by reading code)

### Training baseline
```
vla-scripts/finetune.py
  @draccus.wrap() def finetune(cfg: FinetuneConfig)
    -> run_forward_pass()  [L1 loss on action_head.predict_action(hidden_states)]
```
No `training_step` method. Training is a plain Python for-loop over a DataLoader.
Gradient accumulation and DDP are handled inline. Loss is `torch.nn.L1Loss()`.

### Evaluation baseline
```
experiments/robot/libero/run_libero_eval.py
  @draccus.wrap() def eval_libero(cfg: GenerateConfig)
    -> run_task() -> run_episode() -> get_action() -> get_vla_action()
```
Action chunk of length `NUM_ACTIONS_CHUNK=8` is executed open-loop.

### Action generation path
```
experiments/robot/openvla_utils.py::get_vla_action()
  vla.predict_action(..., action_head=action_head)
  returns List[np.ndarray]  (chunk split into list of 8 actions)
```

### Action head (primary)
```
prismatic/models/action_heads.py::L1RegressionActionHead
  .predict_action(actions_hidden_states)  -> (B, chunk_len, action_dim)
  loss = torch.nn.L1Loss()(ground_truth, predicted)
```
CAPRA must work with this continuous head. No discrete token logits.

---

## 2. Best CAPRA mount points

| What | Where | How |
|---|---|---|
| Mining loop | `experiments/robot/capra/run_capra_mining.py` | Standalone; calls env + rollout modules |
| CAPRA training | `vla-scripts/finetune_capra.py` | Re-use `run_forward_pass` for L_anchor; add KL term |
| CAPRA evaluation | `experiments/robot/capra/run_capra_eval.py` | Mirror `eval_libero`; add metric collection |
| Candidate sampling | `capra/candidate_actions.py` | K stochastic VLA forward passes |
| KL loss injection | `finetune_capra.py` after `run_forward_pass` | `lambda * sum_t w_t * KL(q_hat_t, pi_theta)` |

---

## 3. Baseline files that must stay unmodified

| File | Reason |
|---|---|
| `vla-scripts/finetune.py` | Baseline ablation must run unchanged |
| `experiments/robot/libero/run_libero_eval.py` | Baseline eval reproducibility |
| `experiments/robot/libero/libero_utils.py` | Shared env utility |
| `experiments/robot/openvla_utils.py` | Shared model loading |
| `experiments/robot/robot_utils.py` | Shared action utilities |
| All `prismatic/` files | Core model; must not regress |

---

## 4. Minimum patches needed in Phase 2

### 4a. `prismatic/models/action_heads.py` (optional, 1 method)
Add `score_action_chunk(hidden_states, candidate_actions) -> log_probs`
for KL computation. Additive only; no behaviour change to existing methods.

### 4b. `experiments/robot/openvla_utils.py` (optional, 1 kwarg)
Add optional `temperature` kwarg to `get_vla_action`. Default `0` keeps
baseline deterministic. Required only for stochastic candidate sampling.

### 4c. Nothing else in `prismatic/` needs to change for the CAPRA main pipeline.

---

## 5. Recommended final directory tree

```
openvla-oft/
├── docs/
│   ├── CAPRA.md
│   ├── CONTEXT.md
│   ├── progress_capra.md
│   ├── capra_state.json
│   └── capra_plan.md
│
├── experiments/robot/
│   ├── libero/                     # BASELINE -- unmodified
│   │   ├── run_libero_eval.py
│   │   └── libero_utils.py
│   ├── openvla_utils.py            # BASELINE
│   ├── robot_utils.py              # BASELINE
│   └── capra/                      # ALL CAPRA logic
│       ├── __init__.py
│       ├── capra_config.py
│       ├── env_adapter.py
│       ├── snapshot.py
│       ├── state_api.py
│       ├── object_roles.py
│       ├── task_progress.py
│       ├── signals.py
│       ├── footprint.py
│       ├── equivalence.py
│       ├── candidate_actions.py
│       ├── buffer.py
│       ├── rollout.py
│       ├── precursor.py
│       ├── mining_cache.py
│       ├── build_capra_dataset.py
│       ├── metrics.py
│       ├── procedural_splits.py
│       ├── run_capra_mining.py
│       ├── run_capra_eval.py
│       └── report_utils.py
│
├── vla-scripts/
│   ├── finetune.py                 # BASELINE -- unmodified
│   └── finetune_capra.py           # CAPRA training entry point
│
├── tests/capra/
│   ├── __init__.py
│   ├── test_equivalence.py
│   ├── test_metrics.py
│   ├── test_footprint.py
│   ├── test_precursor.py
│   └── test_smoke_pipeline.py
│
└── scripts/capra/
    ├── mine_capra.sh
    ├── train_capra.sh
    ├── eval_capra.sh
    └── smoke_capra.sh
```

**Note on large artefacts**: rollout caches, mined supervision files, and
checkpoints must NOT be stored inside the repo. Pass `--cache_root` and
`--artifact_root` to all scripts; default to `tmp/` which is git-ignored.

---

## 6. Is `vla-scripts/finetune_capra.py` needed?

Yes. Creating a separate `finetune_capra.py` is the right choice because:

1. The CAPRA loss requires loading both the RLDS dataset (anchor batches)
   and the offline mining cache simultaneously -- the baseline dataloader
   loop does not support this.
2. The training loop must track and log CAPRA vs anchor loss split.
3. Keeping baseline `finetune.py` untouched means ablations (no CAPRA)
   run exactly as before without any flags or code changes.
4. The CAPRA training entry point reuses `run_forward_pass` from
   `finetune.py` via import, so there is no code duplication for the
   anchor loss computation.

---

## 7. Phase 2 first priorities

1. `snapshot.save_snapshot` / `restore_snapshot` -- the entire CF rollout
   infrastructure depends on this. Confirm MuJoCo state API in LIBERO env.
2. `state_api.read_state_signals` -- object poses + topple detection.
3. `rollout.short_cf_rollout` -- once snapshot works, this follows directly.
4. `task_progress.libero_stage_progress` -- BDDL predicate checker.
5. `signals.aggregate_footprint_components` + `footprint.compute_footprint`.
6. Wire into `run_capra_mining.py` and produce first non-empty cache.
7. Implement `finetune_capra.py` training loop with anchor + CAPRA loss.
