"""CAPRA training loss computation.

This module is the only place where the CAPRA KL term is assembled.
It is imported by finetune_capra.py; it never modifies test-time
model structure.

Total loss
----------
  L = L_anchor + lambda * sum_t  w_t * KL(q_hat_t || pi_theta_t)

Terms
-----
  L_anchor  standard L1 regression loss from run_forward_pass
            (computed for EVERY batch regardless of CAPRA activation)

  L_capra   sum over activated timesteps t of:
              w_t * KL(q_hat_t || pi_theta_t)

            where:
              q_hat_t   -- safety target distribution (K,) from mining cache
              pi_theta_t -- policy score distribution over K candidate actions
                           approximated as uniform (or score proxy) because
                           exact continuous likelihood is unavailable
              w_t        -- Delta_t * (1 + rho * R_t)

Continuous-action approximation
---------------------------------
Because L1RegressionActionHead produces a point estimate rather than a
full distribution, we cannot evaluate pi_theta(a_i | h_t) exactly.

We use the CANDIDATE-SET APPROXIMATION:
  pi_theta_t(a_i) ∝ exp(-gamma * ||a_i - a_hat_t||_1)
  normalised over the K candidate actions.

This is a score proxy that:
  - Assigns highest probability to the action closest to the policy's
    predicted action a_hat_t.
  - Falls back to uniform when gamma=0 (identical to uniform prior).
  - Never requires computing a density over the full action space.

If the predicted action a_hat_t is not available in a batch
(e.g. baseline-mode batch without CAPRA records), L_capra = 0.

KL direction
------------
KL(q_hat_t || pi_theta_t) = sum_i q_hat_t[i] * log(q_hat_t[i] / pi_theta_t[i])

Note: KL(q||p) steers p toward q.  We want pi_theta to match q_hat
(the safety target), so we use KL(q_hat || pi_theta) with gradient
through pi_theta only.

"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Policy score distribution (candidate-set approximation)
# ---------------------------------------------------------------------------

def compute_pi_theta(
    candidate_actions: torch.Tensor,   # (K, chunk_len, action_dim)
    predicted_action: torch.Tensor,    # (chunk_len, action_dim)
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute policy score distribution over K candidate actions.

    pi_theta(a_i) ∝ exp(-gamma * ||a_i - a_hat||_1)

    Returns normalised probability vector of shape (K,).
    When gamma == 0 returns uniform distribution.
    """
    if gamma == 0.0:
        K = candidate_actions.shape[0]
        return torch.ones(K, dtype=torch.float32, device=candidate_actions.device) / K

    # L1 distance: (K,)
    l1_dist = (candidate_actions - predicted_action.unsqueeze(0)).abs().sum(dim=(-2, -1))
    log_scores = -gamma * l1_dist
    return F.softmax(log_scores, dim=0)


# ---------------------------------------------------------------------------
# KL divergence (single timestep)
# ---------------------------------------------------------------------------

def kl_q_hat_pi(
    q_hat: torch.Tensor,     # (K,)  safety target distribution
    pi_theta: torch.Tensor,  # (K,)  policy score distribution
    eps: float = 1e-12,
) -> torch.Tensor:
    """KL(q_hat || pi_theta) -- scalar.

    KL(q || p) = sum_i q_i * log(q_i / p_i)

    Only sums over positions where q_hat > 0 (others contribute 0).
    Gradient flows through pi_theta.
    """
    q = q_hat.clamp(min=eps)
    p = pi_theta.clamp(min=eps)
    return (q * (q.log() - p.log())).sum()


# ---------------------------------------------------------------------------
# Per-batch CAPRA loss
# ---------------------------------------------------------------------------

def compute_capra_kl_loss(
    capra_records: List[Dict],         # list of training sample dicts from mining cache
    predicted_actions: torch.Tensor,   # (B, chunk_len, action_dim)  from action head
    device: torch.device,
    gamma: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the CAPRA KL term for one training batch.

    For each record in capra_records that corresponds to a batch element,
    we compute:
      L_capra += w_t * KL(q_hat_t || pi_theta_t)

    The function sums over all activated timesteps in the records list.

    Args:
        capra_records:      List of dicts from build_capra_dataset.record_to_training_sample
                            Each dict contains: 'q_hat' (K,), 'weight' float, 'actions' (K,CL,A)
                            May be empty or have fewer entries than the batch.
        predicted_actions:  Policy predictions for each batch element (B, chunk_len, action_dim).
        device:             Torch device.
        gamma:              Score proxy temperature. 0 = uniform prior.
        eps:                Numerical stability epsilon.

    Returns:
        (capra_loss, metrics_dict)
        capra_loss:   scalar tensor, 0.0 if no activated records
        metrics_dict: {capra_loss, activation_ratio, mean_w_t, mean_delta_t}
    """
    if not capra_records:
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        return zero, {
            "capra_loss": 0.0,
            "activation_ratio": 0.0,
            "mean_w_t": 0.0,
            "mean_delta_t": 0.0,
        }

    total_kl = torch.tensor(0.0, device=device)
    n_activated = 0
    sum_w_t = 0.0
    sum_delta_t = 0.0
    B = predicted_actions.shape[0]

    for i, rec in enumerate(capra_records):
        if i >= B:
            break

        q_hat_np = rec.get("q_hat", None)
        weight    = float(rec.get("weight", 0.0))
        actions_np = rec.get("actions", None)  # (K, CL, A)
        delta_t   = float(rec.get("delta_t", 0.0))

        if q_hat_np is None or actions_np is None:
            continue

        q_hat_t = torch.tensor(q_hat_np, dtype=torch.float32, device=device)

        # Skip if E_t is empty (q_hat is all-zero)
        if q_hat_t.sum() < eps:
            continue

        # Build K candidate tensors
        K, CL, A = actions_np.shape
        cands = torch.tensor(actions_np, dtype=torch.float32, device=device)  # (K, CL, A)

        # Policy prediction for this batch element
        pred_i = predicted_actions[i].float()  # (CL, A)

        # Score distribution over candidates
        pi_theta_t = compute_pi_theta(cands, pred_i, gamma=gamma)  # (K,)

        # KL term
        kl = kl_q_hat_pi(q_hat_t, pi_theta_t, eps=eps)
        total_kl = total_kl + weight * kl

        n_activated += 1
        sum_w_t    += weight
        sum_delta_t += delta_t

    n_total = max(len(capra_records), 1)
    metrics = {
        "capra_loss":       total_kl.item(),
        "activation_ratio": n_activated / n_total,
        "mean_w_t":         sum_w_t  / max(n_activated, 1),
        "mean_delta_t":     sum_delta_t / max(n_activated, 1),
    }
    return total_kl, metrics


# ---------------------------------------------------------------------------
# CAPRABatch: light wrapper for the mining cache reader
# ---------------------------------------------------------------------------

class CAPRADatasetReader:
    """Infinite iterator over CAPRA training samples from the mining cache.

    Yields one sample dict per call to __next__.
    Restarts from the beginning when exhausted.

    This reader is stateless except for the sample list; it does NOT
    maintain per-episode state and is safe to use from a single thread.
    """

    def __init__(
        self,
        cache_root,
        dataset_name: str,
        cfg,
        only_activated: bool = True,
    ):
        from experiments.robot.capra.build_capra_dataset import iter_training_samples
        from pathlib import Path
        self._samples = list(
            iter_training_samples(Path(cache_root), dataset_name, cfg, only_activated)
        )
        self._idx = 0

    def __len__(self) -> int:
        return len(self._samples)

    def is_empty(self) -> bool:
        return len(self._samples) == 0

    def next_batch(self, batch_size: int) -> List[Dict]:
        """Return up to batch_size consecutive samples (wraps around)."""
        if self.is_empty():
            return []
        out = []
        for _ in range(batch_size):
            out.append(self._samples[self._idx % len(self._samples)])
            self._idx += 1
        return out
