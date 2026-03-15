# ===== CAPRA 安全目标分布构建器 (build_capra_dataset.py) =====
#
# 作用
# ----
# 将 mining_cache.py 产出的原始挖掘记录转换为训练样本，
# 核心操作是计算"安全目标分布" q_hat_t。
#
# q_hat_t 的数学定义
# ------------------
#   q_hat_t(a_i) ∝ prior(a_i) * exp(-beta * F_t(a_i))  if a_i in E_t
#                = 0                                      otherwise
#
#   物理含义：在等价集 E_t 中，足迹越小的动作概率越高（类似 softmin）。
#   beta 越大，分布越集中在最安全的动作上。
#   prior 目前是均匀分布（1/K）。
#
# 数值稳定性
# ----------
#   使用 log-sum-exp 技巧避免 exp 下溢/上溢：
#   log_unnorm -= log_unnorm.max() 后再 exp
#
# 输出
# ----
#   iter_training_samples()   逐个 yield 训练样本字典（流式，内存友好）
#   build_full_dataset()      合并所有样本到单个 .npz 文件
#   record_to_training_sample() 单条记录转换
#   每个样本包含：embedding, q_hat, weight(w_t), actions, step, episode_id

"""Convert mined CAPRA cache records into training samples.

Takes CAPRAEpisodeCache files produced by run_capra_mining.py and
builds (observation_embedding, safety_target_distribution, weight)
triples suitable for the CAPRA training loop in finetune_capra.py.

Safety target distribution
--------------------------
q_hat_t(a_i) \u221d prior(a_i) * exp(-beta * F_t(a_i))  if a_i in E_t
            = 0                                       otherwise
(normalised over E_t; zero outside E_t)

Dataset schema
--------------
Each sample is a named-array dict with stable, versioned fields:
  embedding   (D,)  float32  -- frozen VLA visual-language embedding
  q_hat       (K,)  float32  -- safety target distribution
  weight      ()    float32  -- w_t = delta_t * (1 + rho * r_t)
  actions     (K, CL, A) float32  -- candidate actions (optional; for analysis)
  step        ()    int32    -- episode step index
  episode_id  str            -- provenance

Incremental / reproducible updates
-----------------------------------
Dataset files are .npz archives; multiple episodes can be concatenated
by calling build_full_dataset() which writes a single merged archive.
Adding new mined episodes just re-runs build_full_dataset(); old
episode caches are unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.mining.mining_cache import CAPRATimestepRecord, CAPRAEpisodeCache
    from experiments.robot.capra.core.capra_config import CAPRAConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety target distribution builder (pure numpy)
# ---------------------------------------------------------------------------

def build_safety_target_distribution(
    footprint_values: np.ndarray,    # (K,)  F_t(a_i)
    equivalent_indices: np.ndarray,  # variable-length indices of E_t
    prior_weights: np.ndarray,       # (K,)
    beta: float,
) -> np.ndarray:
    """Compute q_hat_t over K candidates.

    q_hat_t(a_i) \u221d prior(a_i) * exp(-beta * F_t(a_i))  if a_i in E_t
                = 0                                         otherwise

    Numerically stable via log-sum-exp.
    Returns normalised float32 vector of shape (K,).
    If equivalent_indices is empty returns all-zero vector
    (CAPRA loss is not triggered for this record).
    """
    K = len(footprint_values)
    q = np.zeros(K, dtype=np.float64)

    if len(equivalent_indices) == 0:
        return q.astype(np.float32)

    log_unnorm = (
        np.log(prior_weights[equivalent_indices].astype(np.float64) + 1e-12)
        - beta * footprint_values[equivalent_indices].astype(np.float64)
    )
    log_unnorm -= log_unnorm.max()   # numerical stability
    unnorm = np.exp(log_unnorm)
    q[equivalent_indices] = unnorm / (unnorm.sum() + 1e-12)
    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# Single record -> training sample
# ---------------------------------------------------------------------------

def record_to_training_sample(
    record: "CAPRATimestepRecord",
    cfg: "CAPRAConfig",
) -> Dict[str, object]:
    """Convert one CAPRATimestepRecord to a training sample dict.

    Returns a flat dict with stable keys (see module docstring).
    Returns None when the record has no embedding (embedding-free
    training mode -- caller must handle).
    """
    q_hat = build_safety_target_distribution(
        footprint_values   = record.footprint_values,
        equivalent_indices = record.equivalent_indices,
        prior_weights      = record.prior_weights,
        beta               = cfg.beta,
    )

    emb = record.observation_embedding
    if emb is None:
        emb = np.zeros(0, dtype=np.float32)  # no embedding; caller decides

    return {
        "embedding":  emb.astype(np.float32),
        "q_hat":      q_hat,
        "weight":     np.float32(record.w_t),
        "actions":    record.candidate_actions.astype(np.float32),
        "step":       np.int32(record.step),
        "episode_id": record.episode_id,
        "delta_t":    np.float32(record.delta_t),
        "p_max":      np.float32(record.p_max),
    }


# ---------------------------------------------------------------------------
# Iterator over all samples in a cache directory
# ---------------------------------------------------------------------------

def iter_training_samples(
    cache_root: Path,
    dataset_name: str,
    cfg: "CAPRAConfig",
    only_activated: bool = True,
) -> Iterator[Dict[str, object]]:
    """Yield training sample dicts from all cached episodes.

    Args:
        cache_root:     Root directory of the mining cache.
        dataset_name:   Sub-directory name (e.g. "libero_spatial").
        cfg:            CAPRAConfig (provides beta for q_hat computation).
        only_activated: If True (default), skip records where E_t is empty.
    """
    from experiments.robot.capra.mining.mining_cache import iter_cache_dir

    n_total = 0
    n_skipped = 0
    for episode_cache in iter_cache_dir(cache_root, dataset_name):
        for record in episode_cache.records:
            if only_activated and len(record.equivalent_indices) == 0:
                n_skipped += 1
                continue
            n_total += 1
            yield record_to_training_sample(record, cfg)

    logger.info(
        "iter_training_samples: yielded %d samples (%d skipped) from %s/%s",
        n_total, n_skipped, cache_root, dataset_name
    )


# ---------------------------------------------------------------------------
# Build merged .npz dataset file
# ---------------------------------------------------------------------------

def build_full_dataset(
    cache_root: Path,
    dataset_name: str,
    cfg: "CAPRAConfig",
    output_path: Optional[Path] = None,
    only_activated: bool = True,
) -> Path:
    """Merge all cached records into a single .npz dataset file.

    The output file contains parallel arrays (one row per activated timestep),
    making it trivial to load into a PyTorch Dataset or numpy array.

    Incremental updates: re-run this function after adding new cache files;
    it always reads the full cache directory.

    Args:
        cache_root:    Root of the mining cache.
        dataset_name:  Sub-directory name.
        cfg:           CAPRAConfig.
        output_path:   Where to write the merged dataset.  Defaults to
                       {cache_root}/{dataset_name}_dataset.npz
        only_activated: If True, skip records with empty E_t.

    Returns:
        Path to the written dataset file.
    """
    if output_path is None:
        output_path = Path(cache_root) / f"{dataset_name}_dataset.npz"

    samples = list(iter_training_samples(cache_root, dataset_name, cfg, only_activated))
    if not samples:
        logger.warning("build_full_dataset: no samples found in %s/%s", cache_root, dataset_name)
        np.savez(output_path, n_samples=np.array(0))
        return output_path

    # Stack each field
    embeddings  = np.stack([s["embedding"]  for s in samples])
    q_hats      = np.stack([s["q_hat"]      for s in samples])
    weights     = np.array([s["weight"]     for s in samples], dtype=np.float32)
    actions     = np.stack([s["actions"]    for s in samples])
    steps       = np.array([s["step"]       for s in samples], dtype=np.int32)
    episode_ids = np.array([s["episode_id"] for s in samples], dtype=object)
    delta_ts    = np.array([s["delta_t"]    for s in samples], dtype=np.float32)
    p_maxes     = np.array([s["p_max"]      for s in samples], dtype=np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        n_samples   = np.array(len(samples), dtype=np.int32),
        embeddings  = embeddings,
        q_hats      = q_hats,
        weights     = weights,
        actions     = actions,
        steps       = steps,
        episode_ids = episode_ids,
        delta_ts    = delta_ts,
        p_maxes     = p_maxes,
        dataset_name= np.array(dataset_name),
        beta        = np.array(cfg.beta, dtype=np.float32),
    )
    logger.info("Dataset written: %s  (%d samples)", output_path, len(samples))
    return output_path


def load_full_dataset(path: Path) -> Dict[str, np.ndarray]:
    """Load a dataset produced by build_full_dataset into a flat dict."""
    data = np.load(path, allow_pickle=True)
    return dict(data)
