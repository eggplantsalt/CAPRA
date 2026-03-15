"""
finetune_capra.py

CAPRA fine-tuning entry point for OpenVLA-OFT.

This script extends the baseline finetune.py training loop with the
CAPRA objective. The anchor loss keeps the policy close to the base;
the CAPRA term steers it toward safer task-equivalent actions.

Total loss:
  L = L_anchor + lambda * sum_t  w_t * KL(q_hat_t || pi_theta)

where:
  L_anchor  = standard L1 regression (continuous action head)
  q_hat_t   = safety target distribution (from offline mining cache)
  w_t       = Delta_t * (1 + rho * R_t)  -- per-timestep weight

Design principles:
  - Baseline finetune.py is NOT modified.
  - All CAPRA logic is imported from experiments/robot/capra/.
  - Only L_anchor is active when no CAPRA supervision exists for a batch.
  - Test-time model structure is identical to baseline OpenVLA-OFT.
  - shuffle_buffer_size defaults to 2000 to avoid OOM on RLDS loader.

Phase 1: config dataclass + argument skeleton.
Phase 2: implement CAPRA loss computation and training loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus

from experiments.robot.capra.capra_config import CAPRAConfig


@dataclass
class FinetuneCAPRAConfig(CAPRAConfig):
    """Training config for CAPRA fine-tuning.

    Inherits all CAPRA hyper-parameters from CAPRAConfig.
    Adds a few training-only knobs not needed in eval/mining.
    """
    # CAPRA loss activation
    capra_warmup_steps: int = 1000   # steps before CAPRA loss is switched on
    log_capra_loss_freq: int = 50    # how often to log CAPRA vs anchor split

    # Resume
    resume: bool = False
    resume_step: Optional[int] = None

    # Checkpoint
    save_latest_checkpoint_only: bool = False
    merge_lora_during_training: bool = True
    val_freq: int = 10_000
    val_time_limit: int = 180
    use_val_set: bool = False
    diffusion_sample_freq: int = 50
    num_diffusion_steps_train: int = 50
    lr_warmup_steps: int = 0
    wandb_log_freq: int = 10


@draccus.wrap()
def finetune_capra(cfg: FinetuneCAPRAConfig) -> None:
    """CAPRA fine-tuning main entry point.

    Phase 2 implementation plan:
      1. Reuse all model loading / LoRA setup from finetune.py.
      2. Load RLDS dataset with anchor batches (same as baseline).
      3. Load CAPRA mining cache (iter_training_samples from build_capra_dataset).
      4. For each batch:
         a. Compute L_anchor via run_forward_pass (imported from finetune.py).
         b. If CAPRA records exist for this batch and step >= capra_warmup_steps:
            Compute CAPRA KL loss weighted by w_t.
         c. Total loss = L_anchor + lambda * L_capra.
         d. Backward + optimizer step.
      5. Log CAPRA / anchor loss split to W&B.
      6. Save checkpoints (same format as baseline -- no new modules at test time).
    """
    raise NotImplementedError(
        "Phase 2: implement CAPRA training loop. "
        "Reuse run_forward_pass from vla-scripts/finetune.py; "
        "add CAPRA KL term from build_capra_dataset.iter_training_samples."
    )


if __name__ == "__main__":
    finetune_capra()
