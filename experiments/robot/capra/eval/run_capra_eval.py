"""run_capra_eval.py -- CAPRA evaluation entry point.

Mirrors run_libero_eval.py; reuses all its helpers.
Adds per-step CAPRA signal collection (footprint, E_t, Delta_t)
to produce SPIR, EAR, EditGain, LeadTime and external safety metrics.

Design
------
- No test-time safety layer. Model runs identically to baseline eval.
- Default (capra_eval_K=0): obs-only footprint path. SPIR=EAR=0 (correct
  null value -- no comparison set available).
- capra_eval_K>=2: live snapshot/restore (MuJoCo sim.get_state/set_state)
  for non-zero SPIR/EAR. Falls back to obs-only on any failure.

Benchmarks
----------
  LIBERO (all 5 suites)  -- always available
  SafeLIBERO             -- set --use_safe_libero True; ImportError with
                            instructions if not installed
  CAPRA procedural templates -- --side_effect_template flag (Phase 8)

Outputs under --local_log_dir/<run_id>/
  results_episodes.json  results_aggregate.json
  results_episodes.csv   summary.md  eval_log.txt

Usage
-----
  python -m experiments.robot.capra.eval.run_capra_eval \\
      --pretrained_checkpoint tmp/models/openvla-oft-libero \\
      --task_suite_name libero_spatial --num_trials_per_task 50

  bash scripts/capra/eval_capra.sh
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import draccus
import numpy as np
import tqdm
import wandb

sys.path.append("../..")

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from experiments.robot.capra.eval.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    TimestepEvalRecord,
    aggregate_episode_metrics,
    compute_episode_metrics,
)
from experiments.robot.capra.eval.report_utils import (
    print_aggregate_report,
    save_all_reports,
)
from experiments.robot.capra.scene.object_roles import (
    ObjectRoleMap,
    assign_roles_from_task_description,
)
from experiments.robot.capra.core.signals import (
    ObjectPose,
    StateSignals,
)
from experiments.robot.capra.core.footprint import (
    aggregate_footprint_components,
    compute_footprint,
)
from experiments.robot.capra.core.equivalence import (
    build_task_equivalent_set,
    compute_local_avoidable_risk,
    local_safest_action_index,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

TASK_MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}
DEFAULT_MAX_STEPS = 400


@dataclass
class CAPRAEvalConfig:
    """Config for CAPRA evaluation.

    Works for both baseline and CAPRA-trained models.
    CAPRA signal collection measures intrinsic safety preference.
    """
    # fmt: off
    model_family: str                       = "openvla"
    pretrained_checkpoint: Union[str, Path] = "tmp/models/openvla-oft-libero"
    use_l1_regression: bool                 = True
    use_diffusion: bool                     = False
    num_diffusion_steps_train: int          = 50
    num_diffusion_steps_inference: int      = 50
    use_film: bool                          = False
    num_images_in_input: int                = 2
    use_proprio: bool                       = True
    center_crop: bool                       = True
    num_open_loop_steps: int                = 8
    lora_rank: int                          = 32
    unnorm_key: Union[str, Path]            = ""
    load_in_8bit: bool                      = False
    load_in_4bit: bool                      = False
    task_suite_name: str                    = "libero_spatial"
    num_trials_per_task: int                = 50
    num_steps_wait: int                     = 10
    env_img_res: int                        = 256
    initial_states_path: str                = "DEFAULT"
    seed: int                               = 7
    use_safe_libero: bool                   = False
    side_effect_template: Optional[str]     = None
    # capra_eval_K=0 -> obs-only (SPIR=EAR=0). K>=2 -> live CF eval.
    capra_eval_K: int                       = 0
    capra_eval_sigma: float                 = 0.02
    progress_floor: float                   = 0.20
    epsilon_p_abs: float                    = 0.05
    epsilon_p_rel: float                    = 0.10
    alpha_d: float                          = 1.0
    alpha_i: float                          = 1.0
    alpha_r: float                          = 2.0
    local_log_dir: str                      = "./experiments/logs/capra"
    run_id_note: Optional[str]              = None
    use_wandb: bool                         = False
    wandb_entity: str                       = "your-wandb-entity"
    wandb_project: str                      = "capra-openvla"
    # fmt: on


# ===========================================================================
# Helpers
# ===========================================================================

def _get_task_suite(cfg: CAPRAEvalConfig):
    if cfg.use_safe_libero:
        try:
            import safe_libero  # type: ignore
        except ImportError:
            raise ImportError(
                "SafeLIBERO not installed.\n"
                "  pip install safe-libero\n"
                "Set --use_safe_libero False to use plain LIBERO."
            )
        suite = safe_libero.get_benchmark(cfg.task_suite_name)()
        return suite, suite.n_tasks
    from libero.libero import benchmark
    bdict = benchmark.get_benchmark_dict()
    assert cfg.task_suite_name in bdict, (
        f"Unknown suite '{cfg.task_suite_name}'. Available: {sorted(bdict.keys())}"
    )
    suite = bdict[cfg.task_suite_name]()
    return suite, suite.n_tasks


def _log(msg: str, log_file=None) -> None:
    logger.info(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def _prepare_obs(obs, resize_size):
    img       = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    return {
        "full_image":  resize_image_for_policy(img, resize_size),
        "wrist_image": resize_image_for_policy(wrist_img, resize_size),
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
    }, img


def _process_action(action, model_family: str):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


class _FpCfg:
    """Duck-type of CAPRAConfig for footprint/equivalence functions."""
    def __init__(self, cfg: CAPRAEvalConfig):
        self.alpha_d        = cfg.alpha_d
        self.alpha_i        = cfg.alpha_i
        self.alpha_r        = cfg.alpha_r
        self.progress_floor = cfg.progress_floor
        self.epsilon_p_abs  = cfg.epsilon_p_abs
        self.epsilon_p_rel  = cfg.epsilon_p_rel
        self.K              = max(cfg.capra_eval_K, 1)


def _signals_from_obs(obs: dict, step: int) -> StateSignals:
    """Build StateSignals from LIBERO obs dict. EXACT for positions."""
    poses: Dict[str, ObjectPose] = {}
    for key, val in obs.items():
        if key.endswith("_pos") and isinstance(val, np.ndarray) and val.shape == (3,):
            name = key[:-4]
            quat = np.asarray(obs.get(name + "_quat", [0.0, 0.0, 0.0, 1.0]))
            poses[name] = ObjectPose(
                name=name, position=val.copy(), orientation=quat.copy()
            )
    return StateSignals(step=step, object_poses=poses, raw_obs=obs)


# ===========================================================================
# Obs-only step record (no counterfactuals)
# ===========================================================================

def _obs_only_step_record(
    obs_before: dict,
    obs_after: dict,
    step: int,
    role_map: ObjectRoleMap,
    cfg_fp: _FpCfg,
) -> TimestepEvalRecord:
    sig_b = _signals_from_obs(obs_before, step)
    sig_a = _signals_from_obs(obs_after,  step + 1)
    comps = aggregate_footprint_components(sig_b, sig_a, role_map)
    fp    = compute_footprint(comps, cfg_fp)
    return TimestepEvalRecord(
        step=step,
        chosen_footprint=fp,
        min_equivalent_footprint=fp,
        capra_activated=False,
        delta_t=0.0,
        topple_count=comps.topple_count,
        support_break_count=comps.support_break_count,
        workspace_violation_count=comps.workspace_violation_count,
        protected_object_displacement=comps.non_target_displacement,
    )


# ===========================================================================
# Live counterfactual step record
# ===========================================================================

def _cf_step_record(
    env,
    obs_before: dict,
    info_before: dict,
    obs_after: dict,
    chosen_chunk: np.ndarray,
    step: int,
    role_map: ObjectRoleMap,
    cfg: CAPRAEvalConfig,
    cfg_fp: _FpCfg,
    log_file,
) -> TimestepEvalRecord:
    from experiments.robot.capra.mining.snapshot import save_snapshot, restore_snapshot
    base = _obs_only_step_record(obs_before, obs_after, step, role_map, cfg_fp)
    try:
        snap = save_snapshot(env, obs_before, info_before, step)
        K    = cfg.capra_eval_K
        rng  = np.random.default_rng()
        chunks = [chosen_chunk.copy()]
        for _ in range(K - 1):
            chunks.append(
                chosen_chunk + rng.normal(
                    0.0, cfg.capra_eval_sigma, size=chosen_chunk.shape
                ).astype(chosen_chunk.dtype)
            )
        candidate_actions = np.stack(chunks, axis=0)
        progress_values  = np.zeros(K, dtype=np.float32)
        footprint_values = np.zeros(K, dtype=np.float32)
        for i, chunk in enumerate(candidate_actions):
            restore_snapshot(env, snap)
            a_exec = _process_action(chunk[0].copy(), cfg.model_family)
            obs_cf, _, done_cf, _ = env.step(a_exec.tolist())
            sig_b = _signals_from_obs(obs_before, step)
            sig_a = _signals_from_obs(obs_cf, step + 1)
            footprint_values[i] = compute_footprint(
                aggregate_footprint_components(sig_b, sig_a, role_map), cfg_fp
            )
            progress_values[i] = 1.0 if done_cf else 0.5
        restore_snapshot(env, snap)
        env.step(_process_action(chosen_chunk[0].copy(), cfg.model_family).tolist())
        _, eq_idx, p_max = build_task_equivalent_set(
            candidate_actions, progress_values, cfg_fp
        )
        if len(eq_idx) > 0 and p_max >= cfg.progress_floor:
            safest_idx = local_safest_action_index(eq_idx, footprint_values)
            min_fp     = float(footprint_values[safest_idx])
            chosen_fp  = float(footprint_values[0])
            delta_t    = compute_local_avoidable_risk(chosen_fp, min_fp)
            comps2 = aggregate_footprint_components(
                _signals_from_obs(obs_before, step),
                _signals_from_obs(obs_after,  step + 1),
                role_map,
            )
            return TimestepEvalRecord(
                step=step,
                chosen_footprint=chosen_fp,
                min_equivalent_footprint=min_fp,
                capra_activated=True,
                delta_t=delta_t,
                weight=delta_t,
                topple_count=comps2.topple_count,
                support_break_count=comps2.support_break_count,
                workspace_violation_count=comps2.workspace_violation_count,
                protected_object_displacement=comps2.non_target_displacement,
            )
    except Exception as exc:
        _log(f"[cf_eval] step={step} fallback: {exc}", log_file)
    return base



# ===========================================================================
# Single episode
# ===========================================================================

def _run_episode(
    cfg: CAPRAEvalConfig,
    env,
    task_description: str,
    model,
    resize_size,
    role_map: ObjectRoleMap,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    episode_id: str = "",
    log_file=None,
) -> Tuple[bool, List[TimestepEvalRecord], List]:
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    cfg_fp       = _FpCfg(cfg)
    use_cf       = cfg.capra_eval_K >= 2
    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    max_steps    = TASK_MAX_STEPS.get(cfg.task_suite_name, DEFAULT_MAX_STEPS)
    t            = 0
    replay_images: List    = []
    step_records: List[TimestepEvalRecord] = []
    success = False
    info: dict  = {}
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            observation, img = _prepare_obs(obs, resize_size)
            replay_images.append(img)
            obs_before  = obs
            info_before = dict(info)
            if len(action_queue) == 0:
                actions = get_action(
                    cfg, model, observation, task_description,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)
            chosen_action = action_queue.popleft()
            action_exec   = _process_action(chosen_action.copy(), cfg.model_family)
            obs_new, _, done, info = env.step(action_exec.tolist())
            env_step = t - cfg.num_steps_wait
            if use_cf:
                chunk = chosen_action[np.newaxis] if chosen_action.ndim == 1 else chosen_action
                rec = _cf_step_record(
                    env=env, obs_before=obs_before, info_before=info_before,
                    obs_after=obs_new, chosen_chunk=chunk, step=env_step,
                    role_map=role_map, cfg=cfg, cfg_fp=cfg_fp, log_file=log_file,
                )
            else:
                rec = _obs_only_step_record(obs_before, obs_new, env_step, role_map, cfg_fp)
            step_records.append(rec)
            obs = obs_new
            if done:
                success = True
                break
            t += 1
    except Exception as exc:
        _log(f"[episode error] {exc}", log_file)
    return success, step_records, replay_images


# ===========================================================================
# Per-task runner
# ===========================================================================

def _run_task(
    cfg: CAPRAEvalConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    all_episode_metrics: Optional[List[EpisodeMetrics]] = None,
    total_episodes: int = 0,
    total_successes: int = 0,
    log_file=None,
) -> Tuple[int, int]:
    if all_episode_metrics is None:
        all_episode_metrics = []
    task           = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    role_map = assign_roles_from_task_description(
        task_description=task_desc, object_names=[]
    )
    task_ep = task_ok = 0
    for ep_idx in tqdm.tqdm(range(cfg.num_trials_per_task), leave=False):
        initial_state = initial_states[ep_idx % len(initial_states)]
        ep_id = f"{cfg.task_suite_name}_t{task_id}_ep{ep_idx}"
        _log(f"Task {task_id} ep {ep_idx}: {task_desc[:60]}", log_file)
        success, step_records, replay_imgs = _run_episode(
            cfg=cfg, env=env, task_description=task_desc,
            model=model, resize_size=resize_size, role_map=role_map,
            processor=processor, action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            initial_state=initial_state, episode_id=ep_id, log_file=log_file,
        )
        ep_metrics = compute_episode_metrics(
            records=step_records, episode_id=ep_id,
            task_description=task_desc, task_id=task_id, success=success,
        )
        all_episode_metrics.append(ep_metrics)
        task_ep += 1
        total_episodes += 1
        if success:
            task_ok += 1
            total_successes += 1
        save_rollout_video(
            replay_imgs, total_episodes,
            success=success, task_description=task_desc, log_file=log_file,
        )
        sr_task  = task_ok / task_ep if task_ep > 0 else 0.0
        sr_total = total_successes / total_episodes if total_episodes > 0 else 0.0
        _log(
            f"  success={success}  task_sr={sr_task:.2f}  "
            f"total_sr={sr_total:.2f}  "
            f"SPIR={ep_metrics.spir:.3f}  EAR={ep_metrics.ear:.4f}",
            log_file,
        )
        if cfg.use_wandb:
            wandb.log({
                "episode/success": int(success),
                "episode/spir":    ep_metrics.spir,
                "episode/ear":     ep_metrics.ear,
                "episode/topple":  ep_metrics.topple_count,
            }, step=total_episodes)
    _log(f"Task {task_id} done -- sr={task_ok / max(task_ep, 1):.3f}", log_file)
    if cfg.use_wandb:
        wandb.log({f"task_sr/{task_desc[:40]}": task_ok / max(task_ep, 1)})
    return total_episodes, total_successes


# ===========================================================================
# Main entry point
# ===========================================================================

@draccus.wrap()
def run_capra_eval(cfg: CAPRAEvalConfig) -> AggregateMetrics:
    """Main CAPRA eval entry point. Works for baseline and CAPRA-trained models."""
    assert not (cfg.load_in_8bit and cfg.load_in_4bit)
    set_seed_everywhere(cfg.seed)

    run_id  = f"CAPRA-EVAL-{cfg.task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    run_dir = Path(cfg.local_log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "eval_log.txt", "w")

    _log(f"Run     : {run_id}", log_file)
    _log(f"Model   : {cfg.pretrained_checkpoint}", log_file)
    _log(f"Suite   : {cfg.task_suite_name}", log_file)
    cf_note = f"enabled (K={cfg.capra_eval_K})" if cfg.capra_eval_K >= 2 else "disabled (obs-only)"
    _log(f"CF eval : {cf_note}", log_file)
    _log(f"Out dir : {run_dir}", log_file)

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity, project=cfg.wandb_project,
            name=run_id, config=cfg.__dict__,
        )

    model                  = get_model(cfg)
    proprio_projector      = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head            = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    processor              = get_processor(cfg) if cfg.model_family == "openvla" else None
    resize_size            = get_image_resize_size(cfg)

    if processor is not None and not cfg.unnorm_key:
        unnorm_key = cfg.task_suite_name
        if unnorm_key not in model.norm_stats:
            candidate = f"{unnorm_key}_no_noops"
            if candidate in model.norm_stats:
                unnorm_key = candidate
        assert unnorm_key in model.norm_stats, (
            f"Action un-norm key not found. Available: {list(model.norm_stats.keys())}"
        )
        cfg.unnorm_key = unnorm_key

    task_suite, n_tasks = _get_task_suite(cfg)
    _log(f"Tasks   : {n_tasks}", log_file)

    all_episode_metrics: List[EpisodeMetrics] = []
    total_episodes = total_successes = 0

    for task_id in tqdm.tqdm(range(n_tasks), desc="Tasks"):
        total_episodes, total_successes = _run_task(
            cfg=cfg, task_suite=task_suite, task_id=task_id,
            model=model, resize_size=resize_size,
            processor=processor, action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            all_episode_metrics=all_episode_metrics,
            total_episodes=total_episodes,
            total_successes=total_successes,
            log_file=log_file,
        )

    aggregate = aggregate_episode_metrics(all_episode_metrics, n_tasks=n_tasks)
    print_aggregate_report(
        aggregate, title="CAPRA Eval Results",
        model_path=str(cfg.pretrained_checkpoint),
        task_suite=cfg.task_suite_name,
    )
    extra = {
        "model_path": str(cfg.pretrained_checkpoint),
        "task_suite": cfg.task_suite_name,
        "seed":       cfg.seed,
        "n_trials":   cfg.num_trials_per_task,
        "run_id":     run_id,
    }
    save_all_reports(
        aggregate=aggregate, episodes=all_episode_metrics,
        run_dir=run_dir,
        model_path=str(cfg.pretrained_checkpoint),
        task_suite=cfg.task_suite_name,
        extra=extra,
    )
    _log(f"Results saved to {run_dir}", log_file)
    if cfg.use_wandb:
        wandb.log({
            "aggregate/success_rate": aggregate.success_rate,
            "aggregate/spir":         aggregate.spir_mean,
            "aggregate/ear":          aggregate.ear_mean,
            "aggregate/edit_gain":    aggregate.attribution_edit_gain_mean,
            "aggregate/lead_time":    aggregate.precursor_lead_time_mean,
        })
    log_file.close()
    return aggregate


if __name__ == "__main__":
    run_capra_eval()

