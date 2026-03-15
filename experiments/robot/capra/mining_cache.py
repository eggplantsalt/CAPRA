"""Offline CAPRA mining cache: schema and I/O.

The mining cache stores per-timestep CAPRA supervision records
produced by run_capra_mining.py.  Each record contains everything
needed to compute the CAPRA training loss without re-running the
simulator at train time.

Cache format
------------
One .npz file per episode, stored under:
  {cache_root}/{dataset_name}/{episode_id}.npz

Naming convention:
  episode_id is sanitised (slashes -> underscores) before use as a filename.

.npz layout
-----------
Scalars / metadata stored as 0-d or 1-d object arrays:
  episode_id        str
  task_description  str
  dataset_name      str
  total_steps       int
  n_activated       int
  n_records         int

Per-record arrays (parallel, length = n_activated):
  rec_steps                 (R,)          int32
  rec_p_max                 (R,)          float32
  rec_delta_t               (R,)          float32
  rec_r_t                   (R,)          float32
  rec_w_t                   (R,)          float32
  rec_candidate_actions     (R, K, CL, A) float32
  rec_prior_weights         (R, K)        float32
  rec_progress_values       (R, K)        float32
  rec_footprint_values      (R, K)        float32
  rec_equivalent_indices_flat  (R*K,)     int32   (ragged -> padded)
  rec_eq_lengths            (R,)          int32   (actual E_t size per record)
  rec_observation_embeddings (R, D)       float32 (D=0 if not captured)
  rec_safest_action_indices (R,)          int32

Resume / incremental update
---------------------------
If the output .npz already exists for an episode, mining skips that
episode unless --force_remine is set.  This makes it safe to rerun
mining scripts after a crash without redoing finished episodes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel for "no embedding captured"
_EMPTY_EMBEDDING_DIM = 0


# ---------------------------------------------------------------------------
# Per-timestep record (rich intermediate form)
# ---------------------------------------------------------------------------

@dataclass
class CAPRATimestepRecord:
    """Supervision data for one activated CAPRA timestep.

    'Activated' means E_t was non-empty (CAPRA loss will be triggered).
    """
    step: int
    candidate_actions: np.ndarray      # (K, chunk_len, action_dim)
    prior_weights: np.ndarray          # (K,)
    progress_values: np.ndarray        # (K,)  P_t(a_i)
    footprint_values: np.ndarray       # (K,)  F_t(a_i)
    equivalent_indices: np.ndarray     # variable-length, indices into K
    p_max: float
    delta_t: float                     # local avoidable risk
    r_t: float = 0.0                   # precursor attribution (Phase 4)
    w_t: float = 0.0                   # loss weight (Phase 4; = delta_t for now)
    safest_action_idx: int = -1        # argmin F_t in E_t
    observation_embedding: Optional[np.ndarray] = None  # (D,)
    task_description: str = ""
    episode_id: str = ""

    def __post_init__(self):
        # Default w_t = delta_t (precursor attribution not yet computed)
        if self.w_t == 0.0 and self.delta_t > 0.0:
            self.w_t = self.delta_t


# ---------------------------------------------------------------------------
# Per-episode cache
# ---------------------------------------------------------------------------

@dataclass
class CAPRAEpisodeCache:
    """All activated CAPRA records for one episode."""
    episode_id: str
    task_description: str
    dataset_name: str
    records: List[CAPRATimestepRecord] = field(default_factory=list)
    total_steps: int = 0
    n_activated: int = 0   # == len(records)

    def append(self, record: CAPRATimestepRecord) -> None:
        self.records.append(record)
        self.n_activated = len(self.records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode_filename(episode_id: str) -> str:
    safe = episode_id.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return safe + ".npz"


def _cache_path(cache_root: Path, dataset_name: str, episode_id: str) -> Path:
    return cache_root / dataset_name / _episode_filename(episode_id)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_episode_cache(cache: CAPRAEpisodeCache, cache_root: Path) -> Path:
    """Serialise one episode's CAPRA records to a .npz file.

    Returns the path of the written file.
    """
    out_path = _cache_path(cache_root, cache.dataset_name, cache.episode_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    R = len(cache.records)
    if R == 0:
        np.savez(
            out_path,
            episode_id       = np.array(cache.episode_id),
            task_description = np.array(cache.task_description),
            dataset_name     = np.array(cache.dataset_name),
            total_steps      = np.array(cache.total_steps, dtype=np.int32),
            n_activated      = np.array(0, dtype=np.int32),
            n_records        = np.array(0, dtype=np.int32),
        )
        return out_path

    # Infer dims from first record
    K       = cache.records[0].candidate_actions.shape[0]
    CL      = cache.records[0].candidate_actions.shape[1]
    A       = cache.records[0].candidate_actions.shape[2]
    emb     = cache.records[0].observation_embedding
    D       = emb.shape[0] if emb is not None else _EMPTY_EMBEDDING_DIM

    # Preallocate
    rec_steps              = np.zeros(R, dtype=np.int32)
    rec_p_max              = np.zeros(R, dtype=np.float32)
    rec_delta_t            = np.zeros(R, dtype=np.float32)
    rec_r_t                = np.zeros(R, dtype=np.float32)
    rec_w_t                = np.zeros(R, dtype=np.float32)
    rec_candidate_actions  = np.zeros((R, K, CL, A), dtype=np.float32)
    rec_prior_weights      = np.zeros((R, K), dtype=np.float32)
    rec_progress_values    = np.zeros((R, K), dtype=np.float32)
    rec_footprint_values   = np.zeros((R, K), dtype=np.float32)
    rec_eq_indices_flat    = np.full(R * K, -1, dtype=np.int32)  # padded with -1
    rec_eq_lengths         = np.zeros(R, dtype=np.int32)
    rec_obs_embeddings     = np.zeros((R, D), dtype=np.float32) if D > 0 else np.zeros((R, 1), dtype=np.float32)
    rec_safest_idx         = np.full(R, -1, dtype=np.int32)

    for i, rec in enumerate(cache.records):
        rec_steps[i]                     = rec.step
        rec_p_max[i]                     = rec.p_max
        rec_delta_t[i]                   = rec.delta_t
        rec_r_t[i]                       = rec.r_t
        rec_w_t[i]                       = rec.w_t
        rec_candidate_actions[i]         = rec.candidate_actions
        rec_prior_weights[i]             = rec.prior_weights
        rec_progress_values[i]           = rec.progress_values
        rec_footprint_values[i]          = rec.footprint_values
        eq = rec.equivalent_indices
        rec_eq_lengths[i]                = len(eq)
        rec_eq_indices_flat[i*K : i*K + len(eq)] = eq
        if rec.observation_embedding is not None and D > 0:
            rec_obs_embeddings[i]        = rec.observation_embedding
        rec_safest_idx[i]                = rec.safest_action_idx

    np.savez(
        out_path,
        episode_id              = np.array(cache.episode_id),
        task_description        = np.array(cache.task_description),
        dataset_name            = np.array(cache.dataset_name),
        total_steps             = np.array(cache.total_steps, dtype=np.int32),
        n_activated             = np.array(R, dtype=np.int32),
        n_records               = np.array(R, dtype=np.int32),
        K                       = np.array(K, dtype=np.int32),
        emb_dim                 = np.array(D, dtype=np.int32),
        rec_steps               = rec_steps,
        rec_p_max               = rec_p_max,
        rec_delta_t             = rec_delta_t,
        rec_r_t                 = rec_r_t,
        rec_w_t                 = rec_w_t,
        rec_candidate_actions   = rec_candidate_actions,
        rec_prior_weights       = rec_prior_weights,
        rec_progress_values     = rec_progress_values,
        rec_footprint_values    = rec_footprint_values,
        rec_eq_indices_flat     = rec_eq_indices_flat,
        rec_eq_lengths          = rec_eq_lengths,
        rec_obs_embeddings      = rec_obs_embeddings,
        rec_safest_idx          = rec_safest_idx,
    )
    logger.info("Saved episode cache: %s (%d activated steps)", out_path, R)
    return out_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_episode_cache(path: Path) -> CAPRAEpisodeCache:
    """Deserialise a CAPRAEpisodeCache from a .npz file."""
    data = np.load(path, allow_pickle=True)

    episode_id       = str(data["episode_id"])
    task_description = str(data["task_description"])
    dataset_name     = str(data["dataset_name"])
    total_steps      = int(data["total_steps"])
    n_records        = int(data["n_records"])

    cache = CAPRAEpisodeCache(
        episode_id=episode_id,
        task_description=task_description,
        dataset_name=dataset_name,
        total_steps=total_steps,
    )

    if n_records == 0:
        return cache

    K              = int(data["K"])
    D              = int(data["emb_dim"])
    eq_flat        = data["rec_eq_indices_flat"]
    eq_lengths     = data["rec_eq_lengths"]
    obs_embeddings = data["rec_obs_embeddings"]

    for i in range(n_records):
        eq_len = int(eq_lengths[i])
        eq_idx = eq_flat[i * K : i * K + eq_len]
        emb = obs_embeddings[i] if D > 0 else None

        rec = CAPRATimestepRecord(
            step                 = int(data["rec_steps"][i]),
            candidate_actions    = data["rec_candidate_actions"][i],
            prior_weights        = data["rec_prior_weights"][i],
            progress_values      = data["rec_progress_values"][i],
            footprint_values     = data["rec_footprint_values"][i],
            equivalent_indices   = eq_idx.astype(np.int32),
            p_max                = float(data["rec_p_max"][i]),
            delta_t              = float(data["rec_delta_t"][i]),
            r_t                  = float(data["rec_r_t"][i]),
            w_t                  = float(data["rec_w_t"][i]),
            safest_action_idx    = int(data["rec_safest_idx"][i]),
            observation_embedding= emb,
            task_description     = task_description,
            episode_id           = episode_id,
        )
        cache.append(rec)

    return cache


# ---------------------------------------------------------------------------
# Directory iterator
# ---------------------------------------------------------------------------

def iter_cache_dir(
    cache_root: Path,
    dataset_name: str,
) -> Iterator[CAPRAEpisodeCache]:
    """Yield CAPRAEpisodeCache objects from all .npz files in a cache dir."""
    cache_dir = Path(cache_root) / dataset_name
    if not cache_dir.exists():
        return
    for path in sorted(cache_dir.glob("*.npz")):
        try:
            yield load_episode_cache(path)
        except Exception as exc:
            logger.warning("Failed to load cache file %s: %s", path, exc)


def list_cached_episode_ids(cache_root: Path, dataset_name: str) -> List[str]:
    """Return list of episode ids that already have a cache file.

    Used for resume / skip-already-mined logic in run_capra_mining.py.
    """
    cache_dir = Path(cache_root) / dataset_name
    if not cache_dir.exists():
        return []
    ids = []
    for path in sorted(cache_dir.glob("*.npz")):
        try:
            data = np.load(path, allow_pickle=True)
            ids.append(str(data["episode_id"]))
        except Exception:
            pass
    return ids
