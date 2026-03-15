"""Candidate action generation for CAPRA.

The candidate set A_t at timestep t is built by running K stochastic
forward passes through the frozen VLA policy.  Each pass produces one
continuous action chunk of shape (chunk_len, action_dim).

Compatibility with OpenVLA-OFT action head
------------------------------------------
OpenVLA-OFT uses L1RegressionActionHead (or DiffusionActionHead).
Both produce deterministic output given the same inputs.  To obtain K
distinct candidates we add small Gaussian noise to the hidden states
before they reach the action head.  This is the "noise-injection"
sampling strategy -- the only change to the forward pass is adding
epsilon ~ N(0, sigma^2) to actions_hidden_states before predict_action.

This avoids any change to the model weights or the action head class.

Interface contract
------------------
  sample_k_action_chunks(vla, processor, obs, task, action_head,
                         proprio_projector, cfg, rng)
      -> np.ndarray  shape (K, chunk_len, action_dim)  -- all in raw space

  uniform_prior_weights(K) -> np.ndarray  shape (K,)

Configuration knobs (all in CAPRAConfig)
-----------------------------------------
  K                  number of candidate chunks to sample  (default 8)
  candidate_noise_sigma  std of Gaussian noise injected into hidden states
                     (default 0.02 -- tunable; 0.0 gives K identical copies)

"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.core.capra_config import CAPRAConfig


# ---------------------------------------------------------------------------
# Prior weights
# ---------------------------------------------------------------------------

def uniform_prior_weights(K: int) -> np.ndarray:
    """Return uniform prior weights over K candidates -- shape (K,).

    This is prior(a_i) in the safety target distribution q_hat_t.
    Replace with likelihood-based weights when a score is available.
    """
    return np.ones(K, dtype=np.float32) / K


# ---------------------------------------------------------------------------
# Single deterministic forward pass (reuses get_vla_action logic)
# ---------------------------------------------------------------------------

def _one_action_chunk(
    vla,
    processor: Any,
    obs: Dict,
    task_description: str,
    action_head,
    proprio_projector,
    cfg: "CAPRAConfig",
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """Run one VLA forward pass and return the action chunk.

    If noise_sigma > 0 we add Gaussian noise to the raw action output
    BEFORE un-normalization.  This is simpler and more stable than
    noise injection into hidden states.

    Returns np.ndarray of shape (chunk_len, action_dim).
    """
    import torch
    from experiments.robot.openvla_utils import get_vla_action

    # get_vla_action returns a list of chunk_len actions (each shape (action_dim,))
    action_list = get_vla_action(
        cfg=cfg,
        vla=vla,
        processor=processor,
        obs=obs,
        task_label=task_description,
        action_head=action_head,
        proprio_projector=proprio_projector,
        use_film=getattr(cfg, "use_film", False),
    )
    chunk = np.stack(action_list, axis=0)   # (chunk_len, action_dim)

    if noise_sigma > 0.0:
        rng_local = np.random.default_rng()
        chunk = chunk + rng_local.normal(0.0, noise_sigma, size=chunk.shape).astype(chunk.dtype)

    return chunk


# ---------------------------------------------------------------------------
# Sample K action chunks
# ---------------------------------------------------------------------------

def sample_k_action_chunks(
    vla,
    processor: Any,
    obs: Dict,
    task_description: str,
    action_head,
    proprio_projector,
    cfg: "CAPRAConfig",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample K action chunks from the VLA policy.

    Strategy: run K forward passes.
      - Pass 0: deterministic (noise_sigma=0) -- this is what the policy
        would actually execute; used as the 'chosen' action.
      - Passes 1..K-1: noise-perturbed (noise_sigma=cfg.candidate_noise_sigma)

    Returns:
        np.ndarray of shape (K, chunk_len, action_dim)

    The first element [0] is always the policy's nominal action.
    """
    K = cfg.K
    noise_sigma = getattr(cfg, "candidate_noise_sigma", 0.02)

    chunks: List[np.ndarray] = []

    for i in range(K):
        sigma = 0.0 if i == 0 else noise_sigma
        chunk = _one_action_chunk(
            vla, processor, obs, task_description,
            action_head, proprio_projector, cfg,
            noise_sigma=sigma,
        )
        chunks.append(chunk)

    return np.stack(chunks, axis=0)   # (K, chunk_len, action_dim)


# ---------------------------------------------------------------------------
# Full candidate set builder
# ---------------------------------------------------------------------------

def build_candidate_set(
    vla,
    processor: Any,
    obs: Dict,
    task_description: str,
    action_head,
    proprio_projector,
    cfg: "CAPRAConfig",
    buffer: Optional[Any] = None,   # SafetyAlternativeBuffer | None (Phase 4)
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the full candidate action set A_t.

    Currently: policy samples only (buffer retrieval is Phase 4).

    Returns:
        actions:       (K, chunk_len, action_dim)
        prior_weights: (K,)  -- uniform
    """
    actions = sample_k_action_chunks(
        vla, processor, obs, task_description,
        action_head, proprio_projector, cfg, rng=rng,
    )
    prior = uniform_prior_weights(len(actions))
    return actions, prior


# ---------------------------------------------------------------------------
# Synthetic candidate generator (no VLA -- for tests and dry runs)
# ---------------------------------------------------------------------------

def synthetic_candidates(
    K: int,
    chunk_len: int = 8,
    action_dim: int = 7,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate K random action chunks for testing without a VLA.

    Returns:
        actions:       (K, chunk_len, action_dim)  -- uniform in [-1, 1]
        prior_weights: (K,)  -- uniform
    """
    if rng is None:
        rng = np.random.default_rng(42)
    actions = rng.uniform(-1.0, 1.0, size=(K, chunk_len, action_dim)).astype(np.float32)
    prior   = uniform_prior_weights(K)
    return actions, prior
