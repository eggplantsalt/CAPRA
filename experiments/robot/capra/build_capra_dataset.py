"""Convert mined CAPRA cache records into training samples.

Takes CAPRAEpisodeCache files produced by run_capra_mining.py and
builds (observation, safety_target_distribution, weight) triples
suitable for the CAPRA training loop in finetune_capra.py.

Safety target distribution per activated timestep:
  q_hat_t(a_i) \u221d prior(a_i) * exp(-beta * F_t(a_i)) * 1[a_i in E_t]
  (normalised over E_t; zero outside E_t)

Phase 1: distribution builder (pure numpy, no env dependency) -- 
  this part is implementable now.
Phase 2: full dataset builder with file I/O.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.mining_cache import CAPRATimestepRecord, CAPRAEpisodeCache
    from experiments.robot.capra.capra_config import CAPRAConfig


def build_safety_target_distribution(
    footprint_values: np.ndarray,    # (K,)
    equivalent_indices: np.ndarray,  # indices of E_t candidates
    prior_weights: np.ndarray,       # (K,)
    beta: float,
) -> np.ndarray:
    """Compute q_hat_t over K candidates.

    q_hat_t(a_i) \u221d prior(a_i) * exp(-beta * F_t(a_i))  if a_i in E_t
                = 0                                         otherwise

    Returns normalised probability vector of shape (K,).
    """
    K = len(footprint_values)
    q = np.zeros(K, dtype=np.float64)

    if len(equivalent_indices) == 0:
        return q  # CAPRA loss not triggered

    log_unnorm = np.log(prior_weights[equivalent_indices] + 1e-12) - beta * footprint_values[equivalent_indices]
    # Numerically stable softmax over equivalent set
    log_unnorm -= log_unnorm.max()
    unnorm = np.exp(log_unnorm)
    q[equivalent_indices] = unnorm / (unnorm.sum() + 1e-12)
    return q.astype(np.float32)


def record_to_training_sample(
    record: "CAPRATimestepRecord",
    cfg: "CAPRAConfig",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert one CAPRATimestepRecord to (embedding, q_hat, weight).

    Returns:
        embedding: (D,)  -- observation embedding
        q_hat: (K,)      -- safety target distribution
        weight: float    -- w_t = Delta_t * (1 + rho * R_t)
    """
    q_hat = build_safety_target_distribution(
        footprint_values=record.footprint_values,
        equivalent_indices=record.equivalent_indices,
        prior_weights=record.prior_weights,
        beta=cfg.beta,
    )
    return record.observation_embedding, q_hat, record.w_t


def iter_training_samples(
    cache_root: Path,
    dataset_name: str,
    cfg: "CAPRAConfig",
) -> Iterator[Tuple[np.ndarray, np.ndarray, float]]:
    """Yield (embedding, q_hat, weight) triples from all cached episodes.

    Phase 2: hook into DataLoader or write to HDF5 for efficient batching.
    """
    from experiments.robot.capra.mining_cache import iter_cache_dir
    for episode_cache in iter_cache_dir(cache_root, dataset_name):
        for record in episode_cache.records:
            yield record_to_training_sample(record, cfg)
