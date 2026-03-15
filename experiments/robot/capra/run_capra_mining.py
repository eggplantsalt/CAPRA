"""Offline CAPRA supervision mining entry point.

Usage
-----
    python -m experiments.robot.capra.run_capra_mining \\
        --pretrained_checkpoint tmp/models/openvla-oft-libero \\
        --dataset_name libero_spatial \\
        --cache_root tmp/capra_cache \\
        --num_mining_episodes 50

Resume / checkpointing
----------------------
If a cache file already exists for an episode, that episode is skipped
unless --force_remine is set.  Safe to interrupt and restart.

Configuration entry points
--------------------------
  K                     MiningConfig.K (default 8)
  H_s                   MiningConfig.H_s (default 5)
  candidate_noise_sigma MiningConfig.candidate_noise_sigma (default 0.02)
  progress_floor        MiningConfig.progress_floor (default 0.20)
  epsilon_p_abs         MiningConfig.epsilon_p_abs (default 0.05)
  epsilon_p_rel         MiningConfig.epsilon_p_rel (default 0.10)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MiningConfig
# ---------------------------------------------------------------------------

@dataclass
class MiningConfig:
    """All knobs for one offline mining run."""
    # VLA
    pretrained_checkpoint: str = "tmp/models/openvla-oft-libero"
    unnorm_key: str = "libero_spatial_no_noops"

    # Dataset / task
    task_suite_name: str = "libero_spatial"
    dataset_name: str = "libero_spatial"

    # Cache
    cache_root: Path = Path("tmp/capra_cache")
    force_remine: bool = False

    # Episode sampling
    num_mining_episodes: int = 50
    max_steps_per_episode: int = 600
    seed: int = 7

    # Environment
    env_img_res: int = 256
    num_images_in_input: int = 2
    use_proprio: bool = True

    # Model
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    lora_rank: int = 32
    center_crop: bool = True

    # CAPRA core knobs
    K: int = 8
    H_s: int = 5
    candidate_noise_sigma: float = 0.02
    progress_floor: float = 0.20
    epsilon_p_abs: float = 0.05
    epsilon_p_rel: float = 0.10
    alpha_d: float = 1.0
    alpha_i: float = 1.0
    alpha_r: float = 2.0
    beta: float = 1.0
    rho: float = 0.5
    lam: float = 0.1

    # SAB
    use_buffer: bool = False
    buffer_max_size: int = 50_000
    buffer_path: Optional[Path] = None

    # Output
    build_merged_dataset: bool = True
    log_every: int = 10

    def to_capra_config(self):
        from experiments.robot.capra.capra_config import CAPRAConfig
        return CAPRAConfig(
            pretrained_checkpoint        = self.pretrained_checkpoint,
            vla_path                     = self.pretrained_checkpoint,
            dataset_name                 = self.dataset_name,
            cache_root                   = self.cache_root,
            K                            = self.K,
            H_s                          = self.H_s,
            candidate_noise_sigma        = self.candidate_noise_sigma,
            progress_floor               = self.progress_floor,
            epsilon_p_abs                = self.epsilon_p_abs,
            epsilon_p_rel                = self.epsilon_p_rel,
            alpha_d                      = self.alpha_d,
            alpha_i                      = self.alpha_i,
            alpha_r                      = self.alpha_r,
            beta                         = self.beta,
            rho                          = self.rho,
            lam                          = self.lam,
            unnorm_key                   = self.unnorm_key,
            use_l1_regression            = self.use_l1_regression,
            use_diffusion                = self.use_diffusion,
            use_film                     = self.use_film,
            use_proprio                  = self.use_proprio,
            num_images_in_input          = self.num_images_in_input,
            center_crop                  = self.center_crop,
            load_in_8bit                 = self.load_in_8bit,
            load_in_4bit                 = self.load_in_4bit,
            num_diffusion_steps_train    = self.num_diffusion_steps_train,
            num_diffusion_steps_inference= self.num_diffusion_steps_inference,
        )


# ---------------------------------------------------------------------------
# Episode miner (env-agnostic core; testable without LIBERO)
# ---------------------------------------------------------------------------

def mine_episode(
    episode_id: str,
    env,
    initial_obs: dict,
    initial_info: dict,
    vla,
    processor,
    action_head,
    proprio_projector,
    role_map,
    cfg_mining: MiningConfig,
    capra_cfg,
    buffer=None,
    progress_fn=None,
):
    """Mine one episode; returns a CAPRAEpisodeCache (not written to disk)."""
    from experiments.robot.capra.mining_cache import (
        CAPRAEpisodeCache, CAPRATimestepRecord,
    )
    from experiments.robot.capra.candidate_actions import (
        build_candidate_set, synthetic_candidates,
    )
    from experiments.robot.capra.snapshot import save_snapshot
    from experiments.robot.capra.rollout import mine_one_timestep
    from experiments.robot.capra.equivalence import local_safest_action_index
    from experiments.robot.capra.buffer import BufferEntry
    import numpy as np

    task_desc = getattr(env, "task_description", "")
    cache = CAPRAEpisodeCache(
        episode_id=episode_id,
        task_description=task_desc,
        dataset_name=cfg_mining.dataset_name,
    )

    obs, info = initial_obs, initial_info

    for step in range(cfg_mining.max_steps_per_episode):
        snap = save_snapshot(
            env, obs=obs, info=info, step=step,
            task_description=task_desc,
        )

        if vla is not None:
            candidate_actions, prior_weights = build_candidate_set(
                vla, processor, obs, task_desc,
                action_head, proprio_projector, capra_cfg, buffer=buffer,
            )
        else:
            candidate_actions, prior_weights = synthetic_candidates(
                capra_cfg.K, rng=np.random.default_rng(step)
            )

        record = mine_one_timestep(
            env, snap, candidate_actions, prior_weights,
            role_map, capra_cfg,
            episode_id=episode_id,
            progress_fn=progress_fn,
        )

        if len(record.equivalent_indices) > 0:
            safest_idx = local_safest_action_index(
                record.equivalent_indices, record.footprint_values
            )
            cache_rec = CAPRATimestepRecord(
                step                 = record.step,
                candidate_actions    = record.candidate_actions,
                prior_weights        = record.prior_weights,
                progress_values      = record.progress_values,
                footprint_values     = record.footprint_values,
                equivalent_indices   = record.equivalent_indices,
                p_max                = record.p_max,
                delta_t              = record.delta_t,
                safest_action_idx    = safest_idx,
                task_description     = task_desc,
                episode_id           = episode_id,
            )
            cache.append(cache_rec)

            if buffer is not None and record.delta_t > 0 and record.obs_embedding is not None:
                buffer.insert(BufferEntry(
                    embedding        = record.obs_embedding,
                    action_chunk     = candidate_actions[safest_idx],
                    footprint        = float(record.footprint_values[safest_idx]),
                    progress         = float(record.progress_values[safest_idx]),
                    task_description = task_desc,
                    source_episode   = episode_id,
                    step             = step,
                ))

        # Step env with nominal action
        nominal = candidate_actions[0]
        done = False
        for a in nominal:
            obs, _, done, info = env.step(a)
            if done:
                break
        if done:
            break

    cache.total_steps = step + 1
    return cache


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_mining(cfg_mining: Optional[MiningConfig] = None) -> None:
    """Main offline mining loop.  Requires LIBERO + trained checkpoint."""
    if cfg_mining is None:
        cfg_mining = MiningConfig()

    from experiments.robot.capra.mining_cache import (
        save_episode_cache, list_cached_episode_ids,
    )
    from experiments.robot.capra.build_capra_dataset import build_full_dataset
    from experiments.robot.capra.buffer import SafetyAlternativeBuffer
    from experiments.robot.capra.object_roles import assign_roles_from_task_description
    from experiments.robot.openvla_utils import (
        get_vla, get_processor, get_action_head, get_proprio_projector,
    )
    from experiments.robot.capra.env_adapter import make_capra_env, EnvConfig

    capra_cfg  = cfg_mining.to_capra_config()
    cache_root = Path(cfg_mining.cache_root)

    vla               = get_vla(capra_cfg)
    processor         = get_processor(capra_cfg)
    action_head       = get_action_head(capra_cfg, llm_dim=vla.llm_dim)
    proprio_projector = (
        get_proprio_projector(capra_cfg, llm_dim=vla.llm_dim, proprio_dim=8)
        if capra_cfg.use_proprio else None
    )

    buffer = None
    if cfg_mining.use_buffer:
        buffer = SafetyAlternativeBuffer(max_size=cfg_mining.buffer_max_size)
        if cfg_mining.buffer_path and Path(cfg_mining.buffer_path).exists():
            buffer.load(Path(cfg_mining.buffer_path))

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[cfg_mining.task_suite_name]()
    tasks = suite.tasks[:cfg_mining.num_mining_episodes]

    cached_ids = set(list_cached_episode_ids(cache_root, cfg_mining.dataset_name))
    logger.info("Resuming: %d episodes already cached", len(cached_ids))

    env_cfg = EnvConfig(
        task_suite_name=cfg_mining.task_suite_name,
        resolution=cfg_mining.env_img_res,
        seed=cfg_mining.seed,
    )

    n_mined = 0
    for task_idx, task in enumerate(tasks):
        episode_id = f"{cfg_mining.dataset_name}__task{task_idx:04d}"

        if episode_id in cached_ids and not cfg_mining.force_remine:
            logger.debug("Skipping already-cached episode: %s", episode_id)
            continue

        env, task_desc = make_capra_env(task, env_cfg)
        env.task_description = task_desc
        obs = env.reset()
        info = {}

        object_names = list(getattr(env._env, "object_names", []))
        role_map = assign_roles_from_task_description(task_desc, object_names)

        cache = mine_episode(
            episode_id=episode_id,
            env=env,
            initial_obs=obs,
            initial_info=info,
            vla=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            role_map=role_map,
            cfg_mining=cfg_mining,
            capra_cfg=capra_cfg,
            buffer=buffer,
        )

        save_episode_cache(cache, cache_root)
        n_mined += 1

        if n_mined % cfg_mining.log_every == 0:
            logger.info(
                "Mined %d/%d episodes | last: %s (%d activated steps)",
                n_mined, len(tasks), episode_id, cache.n_activated,
            )

    logger.info("Mining complete: %d episodes mined", n_mined)

    if cfg_mining.build_merged_dataset:
        from experiments.robot.capra.build_capra_dataset import build_full_dataset
        out = build_full_dataset(cache_root, cfg_mining.dataset_name, capra_cfg)
        logger.info("Merged dataset: %s", out)

    if buffer is not None and cfg_mining.buffer_path:
        buffer.save(Path(cfg_mining.buffer_path))
        logger.info("SAB saved: %s", buffer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_mining()
