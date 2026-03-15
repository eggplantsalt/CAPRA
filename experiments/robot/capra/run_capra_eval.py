"""CAPRA evaluation entry point.

Usage (see scripts/capra/eval_capra.sh):

    python -m experiments.robot.capra.run_capra_eval \\
        --pretrained_checkpoint tmp/models/openvla-oft-libero-capra \\
        --task_suite_name libero_spatial \\
        --num_trials_per_task 50

Outputs per-task and aggregate SPIR, EAR, EditGain, LeadTime,
success rate, protected-object displacement, topple, support-break.

Phase 1: config dataclass + argument parsing skeleton only.
Phase 2: implement eval loop mirroring run_libero_eval.py structure
         with added CAPRA metric collection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus

from experiments.robot.capra.capra_config import CAPRAConfig


@dataclass
class EvalConfig(CAPRAConfig):
    """Extends CAPRAConfig with eval-specific options."""
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 50
    num_steps_wait: int = 10
    env_img_res: int = 256
    initial_states_path: str = "DEFAULT"
    local_log_dir: str = "./experiments/logs/capra"
    seed: int = 7
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_safe_libero: bool = False
    side_effect_template: Optional[str] = None
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "capra-openvla"
    run_id_note: Optional[str] = None


@draccus.wrap()
def run_capra_eval(cfg: EvalConfig) -> None:
    """Main CAPRA evaluation entry point.

    Phase 2: implement eval loop with CAPRA metric collection.
    Baseline LIBERO success rate must still be recorded alongside
    SPIR / EAR so results are directly comparable.
    """
    raise NotImplementedError(
        "Phase 2: implement eval loop with CAPRA metrics. "
        "Mirror run_libero_eval.py structure; add metrics.py calls."
    )


if __name__ == "__main__":
    run_capra_eval()
