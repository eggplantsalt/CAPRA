# ===== CAPRA 微调训练入口 (finetune_capra.py) =====
# 支持两种模式（--capra_enabled False/True）：
#
# Baseline 模式（capra_enabled=False）：
#   与 finetune.py 完全等价，零 CAPRA 开销，用于建立公平对比基线
#
# CAPRA 模式（capra_enabled=True）：
#   L = L_anchor + lambda * sum_t w_t * KL(q_hat_t || pi_theta_t)
#   需要先运行 mine_capra.sh 产出 cache_root 下的挖掘缓存
#
# 关键参数：
#   --shuffle_buffer_size 2000   ⚠️ 必须 <=2000，否则 TF RLDS OOM
#   --capra_warmup_steps 500     预热阶段只跑 anchor loss
#   --lam 0.1                    CAPRA 损失权重
#
# WandB 指标：loss_value, anchor_loss, capra_loss, activation_ratio, mean_delta_t

"""finetune_capra.py -- CAPRA fine-tuning entry point for OpenVLA-OFT.

Total loss:
  L = L_anchor + lambda * sum_t w_t * KL(q_hat_t || pi_theta(.|h_t,l))

Baseline mode (capra_enabled=False):
  Identical to vla-scripts/finetune.py. Zero CAPRA overhead.

CAPRA mode (capra_enabled=True):
  All states   : L_anchor  (L1 regression -- always on)
  Activated only: + lambda * w_t * KL(q_hat_t || pi_theta_t)
    Activation conditions:
      - gradient_step_idx >= capra_warmup_steps
      - capra_records non-empty for this batch
      - q_hat_t non-zero (E_t was non-empty at mining time)

Design notes
------------
* Self-contained: no imports from vla-scripts/finetune.py
  (hyphened directory is not a valid Python package).
* Test-time model structure is NOT modified. CAPRA only changes the loss.
* shuffle_buffer_size=2000 to avoid TF RLDS OOM on remote server.

Training commands
-----------------
  # Baseline:
  torchrun --standalone --nnodes 1 --nproc-per-node 1 \\
      vla-scripts/finetune_capra.py \\
      --vla_path tmp/models/openvla-oft-libero \\
      --dataset_name libero_spatial \\
      --data_root_dir tmp/datasets/rlds \\
      --run_root_dir runs

  # CAPRA:
  torchrun --standalone --nnodes 1 --nproc-per-node 1 \\
      vla-scripts/finetune_capra.py \\
      --vla_path tmp/models/openvla-oft-libero \\
      --dataset_name libero_spatial \\
      --data_root_dir tmp/datasets/rlds \\
      --run_root_dir runs \\
      --capra_enabled True \\
      --cache_root tmp/capra_cache \\
      --lam 0.1 --rho 0.5 --beta 1.0 \\
      --capra_warmup_steps 500
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from experiments.robot.capra.capra_loss import CAPRADatasetReader, compute_capra_kl_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class FinetuneCAPRAConfig:
    """CAPRA fine-tuning config.

    capra_enabled=False => pure baseline, zero overhead.
    capra_enabled=True  => anchor loss + sparse CAPRA KL term.
    """
    # fmt: off
    # ---- model / data ----
    vla_path: str                   = "tmp/models/openvla-oft-libero"
    data_root_dir: Path             = Path("tmp/datasets/rlds")
    dataset_name: str               = "libero_spatial"
    run_root_dir: Path              = Path("runs")

    # ---- architecture ----
    use_l1_regression: bool         = True
    use_diffusion: bool             = False
    num_diffusion_steps_train: int  = 50
    use_film: bool                  = False
    num_images_in_input: int        = 2
    use_proprio: bool               = True

    # ---- training ----
    batch_size: int                 = 8
    learning_rate: float            = 5e-4
    lr_warmup_steps: int            = 0
    num_steps_before_decay: int     = 100_000
    grad_accumulation_steps: int    = 1
    max_steps: int                  = 200_000
    save_freq: int                  = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool                    = False
    resume_step: Optional[int]      = None
    image_aug: bool                 = True
    use_val_set: bool               = False
    val_freq: int                   = 10_000
    val_time_limit: int             = 180
    diffusion_sample_freq: int      = 50

    # NOTE: keep at 2000 -- higher values exhaust server RAM on TF RLDS loader
    shuffle_buffer_size: int        = 2000

    # ---- LoRA ----
    use_lora: bool                  = True
    lora_rank: int                  = 32
    lora_dropout: float             = 0.0
    merge_lora_during_training: bool = True

    # ---- logging ----
    wandb_entity: str               = "your-wandb-entity"
    wandb_project: str              = "capra-openvla"
    run_id_note: Optional[str]      = None
    run_id_override: Optional[str]  = None
    wandb_log_freq: int             = 10

    # ---- CAPRA (only active when capra_enabled=True) ----
    capra_enabled: bool             = False   # False => pure baseline
    lam: float                      = 0.1     # lambda: CAPRA loss weight
    rho: float                      = 0.5     # precursor attribution upweight
    beta: float                     = 1.0     # safety target distribution temperature
    capra_gamma: float              = 1.0     # score proxy temperature (0=uniform prior)
    capra_warmup_steps: int         = 1000    # steps before CAPRA KL term activates
    log_capra_loss_freq: int        = 50      # extra CAPRA W&B log frequency
    cache_root: Path                = Path("tmp/capra_cache")
    # fmt: on


# ===========================================================================
# Self-contained helpers
# ===========================================================================

def _remove_ddp_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix added by DDP."""
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def _get_run_id(cfg: FinetuneCAPRAConfig) -> str:
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    if cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
        return run_id
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.image_aug:
        run_id += "--image_aug"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def _load_module_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    ckpt_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    return _remove_ddp_prefix(torch.load(ckpt_path, weights_only=True, map_location=device))


def _wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused,
               gradient_as_bucket_view=True)


def _count_trainable(module: nn.Module, name: str) -> None:
    n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {n}")


def _init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneCAPRAConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused: bool = False,
) -> DDP:
    module = module_class(**module_args)
    _count_trainable(module, module_name)
    if cfg.resume:
        sd = _load_module_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(sd)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return _wrap_ddp(module, device_id, find_unused)


def _smooth(deques: dict) -> dict:
    return {k: sum(dq) / len(dq) for k, dq in deques.items() if dq and len(dq) > 0}


def _log_wandb(metrics: dict, prefix: str, step: int) -> None:
    log_dict = {}
    for name, value in metrics.items():
        display = "Loss" if name == "loss_value" else name.replace("_", " ").title()
        log_dict[f"{prefix}/{display}"] = value
    wandb.log(log_dict, step=step)


# ===========================================================================
# Anchor forward pass
# ===========================================================================

def _run_anchor_forward(
    vla: nn.Module,
    action_head: Optional[nn.Module],
    noisy_action_projector: Optional[nn.Module],
    proprio_projector: Optional[nn.Module],
    batch: dict,
    device_id: int,
    cfg: FinetuneCAPRAConfig,
    num_patches: int,
) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor]]:
    """Standard anchor/baseline forward pass.

    Returns (loss, metrics_dict, predicted_actions_or_None).
    predicted_actions is returned detached float32 when use_l1_regression=True
    so the CAPRA KL term can use it as a score proxy without double-backprop.
    """
    metrics: Dict[str, float] = {}
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # Diffusion: sample noisy actions for noise-predictor input
    if cfg.use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise = noisy_dict["noise"]
        noisy_actions = noisy_dict["noisy_actions"]
        dte = noisy_dict["diffusion_timestep_embeddings"]
    else:
        noise = noisy_actions = dte = None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if cfg.use_proprio else None,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            noisy_actions=noisy_actions if cfg.use_diffusion else None,
            noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
            diffusion_timestep_embeddings=dte if cfg.use_diffusion else None,
            use_film=cfg.use_film,
        )

    gt_token_ids = batch["labels"][:, 1:].to(device_id)
    curr_mask = get_current_action_mask(gt_token_ids)
    next_mask = get_next_actions_mask(gt_token_ids)
    B = batch["input_ids"].shape[0]
    predicted_actions: Optional[torch.Tensor] = None

    if cfg.use_l1_regression:
        last_h = output.hidden_states[-1]           # (B, seq_len, D)
        text_h = last_h[:, num_patches:-1]           # strip vision patches + EOS
        acts_h = (
            text_h[curr_mask | next_mask]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )                                            # (B, chunk*A, D)
        predicted_actions = action_head.module.predict_action(acts_h)  # (B, chunk, A)
        loss = nn.functional.l1_loss(ground_truth_actions, predicted_actions)
        with torch.no_grad():
            metrics["curr_action_l1_loss"] = nn.functional.l1_loss(
                ground_truth_actions[:, 0], predicted_actions[:, 0].detach()
            ).item()
            metrics["next_actions_l1_loss"] = nn.functional.l1_loss(
                ground_truth_actions[:, 1:], predicted_actions[:, 1:].detach()
            ).item()
        predicted_actions = predicted_actions.detach().float()  # detach for CAPRA KL

    elif cfg.use_diffusion:
        last_h = output.hidden_states[-1]
        text_h = last_h[:, num_patches:-1]
        acts_h = (
            text_h[curr_mask | next_mask]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )
        noise_pred = action_head.module.predict_noise(acts_h).reshape(noise.shape)
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

    else:
        # Discrete token prediction (no action head)
        loss = output.loss
        with torch.no_grad():
            predicted_ids = output.logits[:, num_patches:-1].argmax(dim=2)
            metrics["curr_action_accuracy"] = compute_token_accuracy(
                predicted_ids, gt_token_ids, mask=curr_mask
            ).item()
            metrics["next_actions_accuracy"] = compute_token_accuracy(
                predicted_ids, gt_token_ids, mask=next_mask
            ).item()

    metrics["loss_value"] = loss.item()
    return loss, metrics, predicted_actions


# ===========================================================================
# CAPRA-augmented forward pass
# ===========================================================================

def run_capra_forward_pass(
    vla: nn.Module,
    action_head: Optional[nn.Module],
    noisy_action_projector: Optional[nn.Module],
    proprio_projector: Optional[nn.Module],
    batch: dict,
    action_tokenizer: ActionTokenizer,
    device_id: int,
    cfg: FinetuneCAPRAConfig,
    num_patches: int,
    capra_records: List[Dict],
    gradient_step_idx: int,
) -> Tuple[torch.Tensor, Dict]:
    """Anchor loss + optional CAPRA KL term.

    Anchor loss is ALWAYS computed (all states).
    CAPRA KL is added only when all of:
      - cfg.capra_enabled is True
      - cfg.use_l1_regression is True  (diffusion CAPRA is future work)
      - gradient_step_idx >= cfg.capra_warmup_steps
      - capra_records is non-empty
    """
    anchor_loss, anchor_metrics, predicted_actions = _run_anchor_forward(
        vla=vla,
        action_head=action_head,
        noisy_action_projector=noisy_action_projector,
        proprio_projector=proprio_projector,
        batch=batch,
        device_id=device_id,
        cfg=cfg,
        num_patches=num_patches,
    )

    metrics: Dict = {"anchor_loss": anchor_loss.item()}
    metrics.update(anchor_metrics)

    capra_active = (
        cfg.capra_enabled
        and cfg.use_l1_regression
        and gradient_step_idx >= cfg.capra_warmup_steps
        and len(capra_records) > 0
        and predicted_actions is not None
    )

    if capra_active:
        capra_loss, capra_m = compute_capra_kl_loss(
            capra_records=capra_records,
            predicted_actions=predicted_actions,
            device=torch.device(f"cuda:{device_id}"),
            gamma=cfg.capra_gamma,
        )
        total_loss = anchor_loss + cfg.lam * capra_loss
        metrics.update({
            "capra_loss":       capra_m["capra_loss"],
            "activation_ratio": capra_m["activation_ratio"],
            "mean_w_t":         capra_m["mean_w_t"],
            "mean_delta_t":     capra_m["mean_delta_t"],
            "loss_value":       total_loss.item(),
        })
    else:
        total_loss = anchor_loss
        metrics.update({
            "capra_loss": 0.0,
            "activation_ratio": 0.0,
            "mean_w_t": 0.0,
            "mean_delta_t": 0.0,
        })
        if "loss_value" not in metrics:
            metrics["loss_value"] = anchor_loss.item()

    return total_loss, metrics


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _save_checkpoint(
    cfg: FinetuneCAPRAConfig,
    run_dir: Path,
    log_step: int,
    vla: nn.Module,
    processor,
    proprio_projector: Optional[nn.Module],
    noisy_action_projector: Optional[nn.Module],
    action_head: Optional[nn.Module],
    train_dataset,
    distributed_state: PartialState,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        suffix = f"{log_step}_checkpoint.pt"
    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving checkpoint for step {log_step} at {checkpoint_dir}")
    dist.barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(
                proprio_projector.state_dict(),
                checkpoint_dir / f"proprio_projector--{suffix}",
            )
        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(),
                checkpoint_dir / f"noisy_action_projector--{suffix}",
            )
        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(
                action_head.state_dict(),
                checkpoint_dir / f"action_head--{suffix}",
            )
        if cfg.use_film:
            torch.save(
                vla.module.vision_backbone.state_dict(),
                checkpoint_dir / f"vision_backbone--{suffix}",
            )
    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model at: {checkpoint_dir}")
        dist.barrier()


# ===========================================================================
# Main training loop
# ===========================================================================

@draccus.wrap()
def finetune_capra(cfg: FinetuneCAPRAConfig) -> None:
    """CAPRA fine-tuning main entry point."""
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Set --use_lora True"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot use both L1 regression and diffusion simultaneously."
    )

    cfg.vla_path = cfg.vla_path.rstrip("/")
    run_id = _get_run_id(cfg)
    if cfg.capra_enabled:
        run_id += "+capra"
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # W&B
    if distributed_state.is_main_process:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config=cfg.__dict__,
        )

    # ---- Model loading ----
    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device_id)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    if cfg.use_film:
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        ).to(device_id)

    vla = _wrap_ddp(vla, device_id, find_unused=True)

    # ---- Action head / projectors ----
    proprio_projector = noisy_action_projector = action_head = None
    if cfg.use_proprio:
        proprio_projector = _init_module(
            ProprioProjector, "proprio_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )
    if cfg.use_l1_regression:
        action_head = _init_module(
            L1RegressionActionHead, "action_head", cfg, device_id,
            {"input_dim": vla.module.llm_dim,
             "hidden_dim": vla.module.llm_dim,
             "action_dim": ACTION_DIM},
            to_bf16=True,
        )
    if cfg.use_diffusion:
        action_head = _init_module(
            DiffusionActionHead, "action_head", cfg, device_id,
            {"input_dim": vla.module.llm_dim,
             "hidden_dim": vla.module.llm_dim,
             "action_dim": ACTION_DIM,
             "num_diffusion_steps_train": cfg.num_diffusion_steps_train},
            to_bf16=True,
        )
        noisy_action_projector = _init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim},
        )

    # ---- Patch count ----
    NUM_PATCHES = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # ---- Optimizer ----
    trainable = [p for p in vla.parameters() if p.requires_grad]
    for mod in [action_head, noisy_action_projector, proprio_projector]:
        if mod is not None:
            trainable += [p for p in mod.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    # ---- Dataset ----
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        sampler=None, collate_fn=collator, num_workers=0,
    )

    # ---- CAPRA mining cache reader ----
    capra_reader: Optional[CAPRADatasetReader] = None
    if cfg.capra_enabled:
        try:
            capra_reader = CAPRADatasetReader(
                cfg.cache_root, cfg.dataset_name, cfg, only_activated=True,
            )
            if distributed_state.is_main_process:
                print(f"[CAPRA] Loaded {len(capra_reader)} activated records "
                      f"from {cfg.cache_root}/{cfg.dataset_name}")
        except Exception as exc:
            if distributed_state.is_main_process:
                print(f"[CAPRA] WARNING: could not load mining cache: {exc}")
                print("[CAPRA] Running anchor-only (CAPRA term will be zero).")

    # ---- Metric smoothing deques ----
    tracked_keys = [
        "loss_value", "anchor_loss", "capra_loss",
        "activation_ratio", "mean_w_t", "mean_delta_t",
        "curr_action_l1_loss",
    ]
    recent = {k: deque(maxlen=cfg.grad_accumulation_steps) for k in tracked_keys}

    # ---- Training loop ----
    if distributed_state.is_main_process:
        mode_str = "CAPRA" if cfg.capra_enabled else "baseline"
        print(f"[finetune_capra] Starting {mode_str} training: {run_id}")

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = (
                gradient_step_idx if not cfg.resume
                else cfg.resume_step + gradient_step_idx
            )

            # Fetch CAPRA records for this batch (wraps around cache)
            capra_records: List[Dict] = []
            if capra_reader is not None and not capra_reader.is_empty():
                capra_records = capra_reader.next_batch(cfg.batch_size)

            # Forward pass
            loss, metrics = run_capra_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                cfg=cfg,
                num_patches=NUM_PATCHES,
                capra_records=capra_records,
                gradient_step_idx=gradient_step_idx,
            )

            (loss / cfg.grad_accumulation_steps).backward()

            # Accumulate metrics
            for k, v in metrics.items():
                if k in recent:
                    recent[k].append(v)

            smoothed = _smooth(recent)

            # W&B logging
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                _log_wandb(smoothed, "VLA Train", log_step)
                if cfg.capra_enabled and log_step % cfg.log_capra_loss_freq == 0:
                    wandb.log({
                        "CAPRA/anchor_loss":      smoothed.get("anchor_loss", 0.0),
                        "CAPRA/capra_loss":       smoothed.get("capra_loss", 0.0),
                        "CAPRA/activation_ratio": smoothed.get("activation_ratio", 0.0),
                        "CAPRA/mean_w_t":         smoothed.get("mean_w_t", 0.0),
                        "CAPRA/mean_delta_t":     smoothed.get("mean_delta_t", 0.0),
                    }, step=log_step)

            # LR warmup
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                for pg in optimizer.param_groups:
                    pg["lr"] = original_lr * (0.1 + 0.9 * lr_progress)

            # Gradient step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Checkpoint
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                _save_checkpoint(
                    cfg=cfg, run_dir=run_dir, log_step=log_step,
                    vla=vla, processor=processor,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            if log_step >= cfg.max_steps:
                if distributed_state.is_main_process:
                    print(f"[finetune_capra] Reached max_steps={cfg.max_steps}. Done.")
                break


if __name__ == "__main__":
    finetune_capra()




