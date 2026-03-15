"""CAPRA hyper-parameter dataclass.

All fields are config knobs; defaults are starting-point values only.
Pass overrides via CLI (draccus) or directly in Python.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CAPRAConfig:
    # ------------------------------------------------------------------ paths
    cache_root: Path = Path("tmp/capra_cache")      # where mining artefacts go
    artifact_root: Path = Path("tmp/capra_artifacts")  # large rollout outputs

    # ----------------------------------------------------------- base VLA refs
    pretrained_checkpoint: str = "tmp/models/openvla-oft-libero"  # base ckpt
    vla_path: str = "tmp/models/openvla-oft-libero"               # for finetune_capra
    dataset_name: str = "libero_spatial"                           # RLDS dataset
    data_root_dir: Path = Path("tmp/datasets/rlds")
    run_root_dir: Path = Path("runs")

    # ------------------------------------------------- rollout / mining params
    K: int = 8                  # number of candidate action chunks to sample
    H_s: int = 5                # short counterfactual horizon (steps)
    candidate_noise_sigma: float = 0.02  # std of noise added to action chunks for diversity
    W: int = 10                 # precursor attribution lookback window
    attribution_max_steps: int = 10  # max steps to analyse per dangerous trajectory
    attribution_max_replacements: int = 4  # max replacement rollouts per step
    attribution_rollout_len: int = 8   # H_attr: steps per replacement rollout
    attribution_hazard_threshold: float = 0.10  # min F_t to consider a step 'dangerous'
    num_mining_episodes: int = 50  # episodes to mine per dataset split

    # -------------------------------------------- task-equivalence thresholds
    epsilon_p_abs: float = 0.05   # absolute progress gap threshold
    epsilon_p_rel: float = 0.10   # relative progress gap threshold
    progress_floor: float = 0.20  # min P_max to trigger CAPRA loss

    # ----------------------------------------- footprint component weights
    alpha_d: float = 1.0   # non-target displacement weight
    alpha_i: float = 1.0   # contact impulse weight
    alpha_r: float = 2.0   # irreversible event weight

    # ----------------------------------------------- training loss params
    beta: float = 1.0       # temperature for safety target distribution
    rho: float = 0.5        # precursor attribution upweight factor
    lam: float = 0.1        # lambda: weight of CAPRA loss vs anchor loss

    # ---------------------------------------- training hyperparameters (copy-relevant from FinetuneConfig)
    batch_size: int = 8
    learning_rate: float = 5e-4
    max_steps: int = 200_000
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    save_freq: int = 10_000
    shuffle_buffer_size: int = 2000   # keep low to avoid OOM on RLDS loader
    image_aug: bool = True
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: str = ""

    # ----------------------------------------------- logging
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "capra-openvla"
    use_wandb: bool = False
    run_id_note: Optional[str] = None
