"""Offline CAPRA supervision mining entry point.

Usage (see scripts/capra/mine_capra.sh for a ready-made wrapper):

    python -m experiments.robot.capra.run_capra_mining \\
        --pretrained_checkpoint tmp/models/openvla-oft-libero \\
        --dataset_name libero_spatial \\
        --cache_root tmp/capra_cache \\
        --num_mining_episodes 50

What this script does (Phase 2 implementation):
  1. Load base VLA policy.
  2. For each episode in the task suite:
     a. Run a baseline rollout, saving a Snapshot at each step.
     b. At each step, sample K candidate action chunks.
     c. For each candidate, run a short CF rollout to get P_t and F_t.
     d. Build E_t; compute Delta_t; compute R_t via precursor attribution.
     e. Write CAPRAEpisodeCache to cache_root.

Phase 1: config dataclass + argument parsing skeleton only.
Phase 2: implement the mining loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus

from experiments.robot.capra.capra_config import CAPRAConfig


@dataclass
class MiningConfig(CAPRAConfig):
    """Extends CAPRAConfig with mining-specific options."""
    task_suite_name: str = "libero_spatial"
    num_mining_episodes: int = 50
    num_trials_per_task: int = 5
    env_img_res: int = 256
    seed: int = 7
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 32
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_wandb: bool = False


@draccus.wrap()
def run_mining(cfg: MiningConfig) -> None:
    """Main mining entry point.

    Phase 2: implement mining loop.
    """
    raise NotImplementedError(
        "Phase 2: implement mining loop using env_adapter, snapshot, "
        "candidate_actions, rollout, equivalence, precursor, mining_cache."
    )


if __name__ == "__main__":
    run_mining()
