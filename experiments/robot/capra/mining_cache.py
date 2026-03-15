"""Offline CAPRA mining cache: schema and I/O.

The mining cache stores per-timestep CAPRA supervision records
produced by run_capra_mining.py. Each record contains everything
needed to compute the CAPRA training loss without re-running the
simulator at train time.

Cache format: one .npz file per episode, stored under:
  {cache_root}/{dataset_name}/{episode_id}.npz

Phase 1: schema dataclasses + stub I/O.
Phase 2: implement save/load with numpy archives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CAPRATimestepRecord:
    """Supervision data for one activated CAPRA timestep."""
    step: int
    observation_embedding: np.ndarray     # frozen VLA embedding  (D,)
    candidate_actions: np.ndarray         # (K, chunk_len, action_dim)
    progress_values: np.ndarray           # (K,)
    footprint_values: np.ndarray          # (K,)
    equivalent_indices: np.ndarray        # indices into candidates
    p_max: float
    delta_t: float                        # local avoidable risk
    r_t: float                            # precursor attribution score
    w_t: float                            # loss weight
    prior_weights: np.ndarray             # (K,)  prior(a_i)
    task_description: str = ""
    episode_id: str = ""


@dataclass
class CAPRAEpisodeCache:
    """All activated CAPRA records for one episode."""
    episode_id: str
    task_description: str
    dataset_name: str
    records: List[CAPRATimestepRecord] = field(default_factory=list)
    total_steps: int = 0
    n_activated: int = 0    # steps where CAPRA loss was triggered


def save_episode_cache(cache: CAPRAEpisodeCache, cache_root: Path) -> Path:
    """Serialise one episode's CAPRA records to disk.

    Returns path to the written file.
    Phase 2: implement numpy archive serialisation.
    """
    raise NotImplementedError("Phase 2.")


def load_episode_cache(path: Path) -> CAPRAEpisodeCache:
    """Deserialise a CAPRAEpisodeCache from disk."""
    raise NotImplementedError("Phase 2.")


def iter_cache_dir(cache_root: Path, dataset_name: str):
    """Yield CAPRAEpisodeCache objects from all files in a cache directory."""
    cache_dir = cache_root / dataset_name
    if not cache_dir.exists():
        return
    for path in sorted(cache_dir.glob("*.npz")):
        yield load_episode_cache(path)
