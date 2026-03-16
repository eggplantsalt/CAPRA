# finetune_capra_deepspeed.py -- DeepSpeed version of CAPRA fine-tuning
#
# Changes vs finetune_capra.py:
#   1. deepspeed.initialize() replaces DDP + AdamW
#   2. deepspeed launcher replaces torchrun
#   3. shuffle_buffer_size asserted <= 2000
#   4. All CAPRA logic unchanged
#
# Launch (single GPU):
#   deepspeed --num_gpus 1 vla-scripts/finetune_capra_deepspeed.py \
#       --vla_path tmp/models/openvla-7b \
#       --dataset_name libero_spatial_no_noops \
#       --data_root_dir tmp/datasets/rlds \
#       --run_root_dir runs \
#       --capra_enabled True \
#       --cache_root tmp/capra_cache \
#       --deepspeed_config vla-scripts/ds_config_zero2.json
#
# Launch (multi-GPU, e.g. 4):
#   deepspeed --num_gpus 4 vla-scripts/finetune_capra_deepspeed.py ...
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import deepspeed
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
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
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from experiments.robot.capra.capra_loss import CAPRADatasetReader, compute_capra_kl_loss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneCAPRADeepSpeedConfig:
    # fmt: off
    vla_path: str                     = "tmp/models/openvla-oft-libero"
    data_root_dir: Path               = Path("tmp/datasets/rlds")
    dataset_name: str                 = "libero_spatial_no_noops"
    run_root_dir: Path                = Path("runs")
    use_l1_regression: bool           = True
    use_diffusion: bool               = False
    num_diffusion_steps_train: int    = 50
    use_film: bool                    = False
    num_images_in_input: int          = 2
    use_proprio: bool                 = True
    batch_size: int                   = 8
    learning_rate: float              = 5e-4
    num_steps_before_decay: int       = 100_000
    grad_accumulation_steps: int      = 1
    max_steps: int                    = 200_000
    save_freq: int                    = 10_000
    save_latest_checkpoint_only: bool = False
    image_aug: bool                   = True
    shuffle_buffer_size: int          = 2000  # must stay <= 2000
    use_lora: bool                    = True
    lora_rank: int                    = 32
    lora_dropout: float               = 0.0
    merge_lora_during_training: bool  = True
    use_wandb: bool                   = False
    wandb_entity: str                 = "your-wandb-entity"
    wandb_project: str                = "capra-openvla"
    run_id_note: Optional[str]        = None
    run_id_override: Optional[str]    = None
    wandb_log_freq: int               = 10
    deepspeed_config: str             = "vla-scripts/ds_config_zero2.json"
    local_rank: int                   = -1
    capra_enabled: bool               = False
    lam: float                        = 0.1
    rho: float                        = 0.5
    beta: float                       = 1.0
    capra_gamma: float                = 1.0
    capra_warmup_steps: int           = 1000
    log_capra_loss_freq: int          = 50
    cache_root: Path                  = Path("tmp/capra_cache")
    # fmt: on


def _get_run_id(cfg):
    if cfg.run_id_override:
        return cfg.run_id_override
    run_id = f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+b{cfg.batch_size}+lr-{cfg.learning_rate}"
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.image_aug:
        run_id += "--image_aug"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def _smooth(deques):
    return {k: sum(dq) / len(dq) for k, dq in deques.items() if dq}


def _is_main():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _anchor_forward(ds_engine, action_head, noisy_ap, proprio_p, batch, device_id, cfg, num_patches):
    metrics = {}
    gt = batch["actions"].to(device_id).to(torch.bfloat16)
    noise = noisy_acts = dte = None
    if cfg.use_diffusion:
        nd = action_head.sample_noisy_actions(gt)
        noise, noisy_acts, dte = nd["noise"], nd["noisy_actions"], nd["diffusion_timestep_embeddings"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = ds_engine(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if cfg.use_proprio else None,
            proprio_projector=proprio_p if cfg.use_proprio else None,
            noisy_actions=noisy_acts if cfg.use_diffusion else None,
            noisy_action_projector=noisy_ap if cfg.use_diffusion else None,
            diffusion_timestep_embeddings=dte if cfg.use_diffusion else None,
            use_film=cfg.use_film,
        )
    gt_ids = batch["labels"][:, 1:].to(device_id)
    curr_mask = get_current_action_mask(gt_ids)
    next_mask = get_next_actions_mask(gt_ids)
    B = batch["input_ids"].shape[0]
    predicted = None
    if cfg.use_l1_regression:
        text_h = out.hidden_states[-1][:, num_patches:-1]
        acts_h = text_h[curr_mask | next_mask].reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
        predicted = action_head.predict_action(acts_h)
        loss = nn.functional.l1_loss(gt, predicted)
        metrics["curr_action_l1_loss"] = nn.functional.l1_loss(gt[:, 0], predicted[:, 0].detach()).item()
        predicted = predicted.detach().float()
    elif cfg.use_diffusion:
        text_h = out.hidden_states[-1][:, num_patches:-1]
        acts_h = text_h[curr_mask | next_mask].reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
        np_out = action_head.predict_noise(acts_h).reshape(noise.shape)
        loss = nn.functional.mse_loss(np_out, noise)
    else:
        loss = out.loss
    metrics["loss_value"] = loss.item()
    return loss, metrics, predicted


def run_forward(ds_engine, action_head, noisy_ap, proprio_p, batch, device_id, cfg, num_patches, capra_records, step_idx):
    anchor_loss, anchor_m, predicted = _anchor_forward(
        ds_engine, action_head, noisy_ap, proprio_p, batch, device_id, cfg, num_patches)
    metrics = {"anchor_loss": anchor_loss.item()}
    metrics.update(anchor_m)
    active = (cfg.capra_enabled and cfg.use_l1_regression
              and step_idx >= cfg.capra_warmup_steps
              and len(capra_records) > 0 and predicted is not None)
    if active:
        cl, cm = compute_capra_kl_loss(
            capra_records=capra_records, predicted_actions=predicted,
            device=torch.device(f"cuda:{device_id}"), gamma=cfg.capra_gamma)
        total = anchor_loss + cfg.lam * cl
        metrics.update({"capra_loss": cm["capra_loss"], "activation_ratio": cm["activation_ratio"],
                        "mean_w_t": cm["mean_w_t"], "mean_delta_t": cm["mean_delta_t"],
                        "loss_value": total.item()})
    else:
        total = anchor_loss
        metrics.update({"capra_loss": 0.0, "activation_ratio": 0.0, "mean_w_t": 0.0, "mean_delta_t": 0.0})
        metrics.setdefault("loss_value", anchor_loss.item())
    return total, metrics


def _save_checkpoint(cfg, run_dir, log_step, ds_engine, processor,
                     proprio_p, noisy_ap, action_head, train_dataset):
    if cfg.save_latest_checkpoint_only:
        ckpt_dir = run_dir
        suffix = "latest_checkpoint.pt"
    else:
        ckpt_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        suffix = f"{log_step}_checkpoint.pt"
    adapter_dir = ckpt_dir / "lora_adapter"
    if _is_main():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, ckpt_dir)
        print(f"Saving checkpoint step={log_step} at {ckpt_dir}")
    dist.barrier()
    # DeepSpeed checkpoint (optimizer state, ZeRO shards)
    ds_engine.save_checkpoint(str(ckpt_dir), tag=f"step_{log_step}")
    if _is_main():
        processor.save_pretrained(ckpt_dir)
        ds_engine.module.save_pretrained(adapter_dir)
        if cfg.use_proprio and proprio_p is not None:
            torch.save(proprio_p.state_dict(), ckpt_dir / f"proprio_projector--{suffix}")
        if cfg.use_diffusion and noisy_ap is not None:
            torch.save(noisy_ap.state_dict(), ckpt_dir / f"noisy_action_projector--{suffix}")
        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), ckpt_dir / f"action_head--{suffix}")
    dist.barrier()
    if cfg.use_lora and cfg.merge_lora_during_training and _is_main():
        base = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True)
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.save_pretrained(ckpt_dir)
        print(f"Merged LoRA saved: {ckpt_dir}")
    dist.barrier()


@draccus.wrap()
def finetune_capra_deepspeed(cfg: FinetuneCAPRADeepSpeedConfig) -> None:
    assert cfg.use_lora, "Only LoRA fine-tuning supported. Set --use_lora True"
    assert not (cfg.use_l1_regression and cfg.use_diffusion)
    assert cfg.shuffle_buffer_size <= 2000, (
        f"shuffle_buffer_size={cfg.shuffle_buffer_size} > 2000 will OOM on TF RLDS loader. "
        "Keep it at 2000 or below.")

    cfg.vla_path = cfg.vla_path.rstrip("/")
    run_id = _get_run_id(cfg)
    if cfg.capra_enabled:
        run_id += "+capra"
    run_id += "+deepspeed"
    run_dir = cfg.run_root_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # DeepSpeed init (also inits distributed)
    deepspeed.init_distributed()
    device_id = int(os.environ.get("LOCAL_RANK", cfg.local_rank))
    if device_id == -1:
        device_id = 0
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if _is_main() and cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                   name=run_id, config=cfg.__dict__)

    # Model
    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    if _is_main():
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA
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
            vision_backbone=vla.model.vision_backbone, llm_dim=vla.llm_dim)

    # Action head / projectors (plain nn.Module, moved to GPU)
    proprio_p = noisy_ap = action_head = None
    if cfg.use_proprio:
        proprio_p = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM).to(device_id)
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM
        ).to(torch.bfloat16).to(device_id)
    if cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM,
            num_diffusion_steps_train=cfg.num_diffusion_steps_train,
        ).to(torch.bfloat16).to(device_id)
        noisy_ap = NoisyActionProjector(llm_dim=vla.llm_dim).to(device_id)

    NUM_PATCHES = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Collect all trainable params for DeepSpeed optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    for mod in [action_head, noisy_ap, proprio_p]:
        if mod is not None:
            trainable_params += [p for p in mod.parameters() if p.requires_grad]

    ds_config = {
        "train_micro_batch_size_per_gpu": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accumulation_steps,
        "optimizer": {"type": "AdamW",
                      "params": {"lr": cfg.learning_rate, "betas": [0.9, 0.999],
                                 "eps": 1e-8, "weight_decay": 0.0}},
        "scheduler": {"type": "WarmupDecayLR",
                      "params": {"warmup_min_lr": 0, "warmup_max_lr": cfg.learning_rate,
                                 "warmup_num_steps": cfg.lr_warmup_steps,
                                 "total_num_steps": cfg.max_steps}},
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
    }
    # Merge with external config file if provided
    import json
    if cfg.deepspeed_config and os.path.exists(cfg.deepspeed_config):
        with open(cfg.deepspeed_config) as f:
            ext = json.load(f)
        # External file takes precedence for zero_optimization / activation_checkpointing
        for k, v in ext.items():
            if k not in ("train_micro_batch_size_per_gpu", "gradient_accumulation_steps",
                         "optimizer", "scheduler"):
                ds_config[k] = v

    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=vla,
        model_parameters=trainable_params,
        config=ds_config,
    )

    # Dataset
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=cfg.num_images_in_input > 1,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name, batch_transform,
        resize_resolution=tuple(ds_engine.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if _is_main():
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

    # CAPRA cache
    capra_reader = None
    if cfg.capra_enabled:
        try:
            capra_reader = CAPRADatasetReader(
                cfg.cache_root, cfg.dataset_name, cfg, only_activated=True)
            if _is_main():
                print(f"[CAPRA] Loaded {len(capra_reader)} records from {cfg.cache_root}/{cfg.dataset_name}")
        except Exception as exc:
            if _is_main():
                print(f"[CAPRA] WARNING: {exc} -- running anchor-only")

    tracked = ["loss_value", "anchor_loss", "capra_loss", "activation_ratio",
               "mean_w_t", "mean_delta_t", "curr_action_l1_loss"]
    recent = {k: deque(maxlen=cfg.grad_accumulation_steps) for k in tracked}

    if _is_main():
        mode = "CAPRA" if cfg.capra_enabled else "baseline"
        print(f"[finetune_capra_deepspeed] Starting {mode} training: {run_id}")

    ds_engine.train()
    with tqdm.tqdm(total=cfg.max_steps, leave=False, disable=not _is_main()) as bar:
        for batch_idx, batch in enumerate(dataloader):
            grad_step = batch_idx // cfg.grad_accumulation_steps
            log_step = grad_step

            capra_records = []
            if capra_reader is not None and not capra_reader.is_empty():
                capra_records = capra_reader.next_batch(cfg.batch_size)

            loss, metrics = run_forward(
                ds_engine, action_head, noisy_ap, proprio_p,
                batch, device_id, cfg, NUM_PATCHES, capra_records, grad_step,
            )

            # DeepSpeed handles gradient accumulation and backward internally
            ds_engine.backward(loss)
            ds_engine.step()

            for k, v in metrics.items():
                if k in recent:
                    recent[k].append(v)
            smoothed = _smooth(recent)

            if _is_main() and log_step % cfg.wandb_log_freq == 0 and cfg.use_wandb:
                wandb.log({f"train/{k}": v for k, v in smoothed.items()}, step=log_step)
                if cfg.capra_enabled and log_step % cfg.log_capra_loss_freq == 0:
                    wandb.log({
                        "CAPRA/anchor_loss": smoothed.get("anchor_loss", 0.0),
                        "CAPRA/capra_loss": smoothed.get("capra_loss", 0.0),
                        "CAPRA/activation_ratio": smoothed.get("activation_ratio", 0.0),
                    }, step=log_step)

            if grad_step > 0 and log_step % cfg.save_freq == 0:
                _save_checkpoint(cfg, run_dir, log_step, ds_engine, processor,
                                 proprio_p, noisy_ap, action_head, train_dataset)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                bar.update()

            if log_step >= cfg.max_steps:
                if _is_main():
                    print(f"[finetune_capra_deepspeed] Reached max_steps={cfg.max_steps}. Done.")
                break


if __name__ == "__main__":
    finetune_capra_deepspeed()
