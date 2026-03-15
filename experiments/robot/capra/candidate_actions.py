"""Candidate action generation for CAPRA.

At each timestep t the candidate set A_t is built from two sources:
  1. Sample-K action chunks from the current policy (primary source).
  2. Retrieved entries from the Safety Alternative Buffer (training only).

The interface is intentionally thin so that the prior can later be
replaced with a sample-score or likelihood proxy without touching
the rest of the pipeline.

Phase 1: interface + sample-K stub.
Phase 2: integrate VLA forward pass + buffer retrieval.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.capra_config import CAPRAConfig


def sample_k_action_chunks(
    vla,
    processor: Any,
    observation: Dict,
    task_description: str,
    action_head: Any,
    proprio_projector: Any,
    cfg: "CAPRAConfig",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample K action chunks from the VLA policy via stochastic forward passes.

    Returns array of shape (K, chunk_len, action_dim).

    Phase 2: run VLA with temperature > 0 or noise injection K times.
    Currently raises NotImplementedError to keep Phase 1 clean.
    """
    raise NotImplementedError(
        "Phase 2: run K stochastic VLA forward passes and stack results."
    )


def uniform_prior_weights(K: int) -> np.ndarray:
    """Return uniform prior weights over K candidates.

    This is the default prior(a_i) used in the safety target distribution
    q_hat_t. Can be replaced with likelihood-based weights in Phase 3.
    """
    return np.ones(K, dtype=np.float32) / K


def build_candidate_set(
    vla,
    processor: Any,
    observation: Dict,
    task_description: str,
    action_head: Any,
    proprio_projector: Any,
    cfg: "CAPRAConfig",
    buffer: Optional[Any] = None,   # SafetyAlternativeBuffer | None
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the full candidate action set A_t.

    Returns:
        actions: (K_total, chunk_len, action_dim)
        prior_weights: (K_total,)  -- uniform unless buffer provides scores

    Phase 2: combine policy samples with buffer retrievals.
    """
    raise NotImplementedError(
        "Phase 2: call sample_k_action_chunks + buffer.retrieve if buffer is not None."
    )
