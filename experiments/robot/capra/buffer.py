"""Safety Alternative Buffer (SAB) -- training-time only.

The SAB stores (embedding, action_chunk) pairs that were identified as
low-footprint, task-equivalent alternatives to higher-risk actions taken
by the base policy.  It is used exclusively during offline mining and
CAPRA dataset building.

CRITICAL: This module MUST NOT be imported from any test-time code path
(deploy.py, run_libero_eval.py, or anything under prismatic/).  The buffer
lives only in the mining pipeline.

Key / value structure
---------------------
Key   embedding: np.ndarray (D,)  -- frozen VLA visual-language embedding
                 concatenated with optional geometric summary (K_geo,)
                 Final key shape: (D + K_geo,)

Value action_chunk: np.ndarray (chunk_len, action_dim)
      footprint: float
      progress:  float
      task_description: str
      source_episode:   str
      step:             int

Retrieval
---------
Default: brute-force L2 nearest neighbour (fast for < 50k entries).
If the buffer exceeds the brute-force threshold, a FAISS flat index is
used automatically when faiss is importable; otherwise stays brute-force.

Persistence
-----------
Save: numpy .npz archive; each array field stored separately.
Load: reconstruct BufferEntry list from the .npz.
Format is human-inspectable and Git-LFS-friendly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Threshold above which we try FAISS for faster retrieval
_FAISS_THRESHOLD = 10_000


# ---------------------------------------------------------------------------
# Buffer entry
# ---------------------------------------------------------------------------

@dataclass
class BufferEntry:
    """One record in the Safety Alternative Buffer.

    Key  : embedding  (D,)  -- used for nearest-neighbour retrieval
    Value: everything else
    """
    embedding: np.ndarray      # (D,) retrieval key
    action_chunk: np.ndarray   # (chunk_len, action_dim) safe alternative
    footprint: float           # F_t of this action
    progress: float            # P_t of this action
    task_description: str = ""
    source_episode: str = ""
    step: int = 0


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class SafetyAlternativeBuffer:
    """In-memory SAB with L2 nearest-neighbour retrieval.

    Interface contract (must not change even when FAISS is added):
      insert(entry)              -> None
      retrieve(query, top_k)     -> List[BufferEntry]
      save(path)                 -> None
      load(path)                 -> None  (classmethod alternative: from_file)
      __len__()                  -> int
    """

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self._entries: List[BufferEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SafetyAlternativeBuffer(size={len(self)}/{self.max_size})"

    # ---------------------------------------------------------------- insert

    def insert(self, entry: BufferEntry) -> None:
        """Add an entry; evict oldest (FIFO) if at capacity."""
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)
        self._entries.append(entry)

    # --------------------------------------------------------------- retrieve

    def retrieve(
        self,
        query_embedding: np.ndarray,  # (D,)
        top_k: int = 4,
    ) -> List[BufferEntry]:
        """Return the top_k entries nearest to query_embedding (L2).

        Returns empty list when the buffer is empty.
        Clamps top_k to the current buffer size.
        """
        n = len(self._entries)
        if n == 0:
            return []
        top_k = min(top_k, n)

        if n >= _FAISS_THRESHOLD:
            return self._retrieve_faiss(query_embedding, top_k)
        return self._retrieve_brute(query_embedding, top_k)

    def _retrieve_brute(self, query: np.ndarray, top_k: int) -> List[BufferEntry]:
        embeddings = np.stack([e.embedding for e in self._entries])  # (N, D)
        dists = np.linalg.norm(embeddings - query[None], axis=1)
        top_idx = np.argsort(dists)[:top_k]
        return [self._entries[i] for i in top_idx]

    def _retrieve_faiss(self, query: np.ndarray, top_k: int) -> List[BufferEntry]:
        try:
            import faiss  # optional dependency
            embeddings = np.stack([e.embedding for e in self._entries]).astype(np.float32)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            _, top_idx = index.search(query[None].astype(np.float32), top_k)
            return [self._entries[i] for i in top_idx[0]]
        except ImportError:
            logger.debug("faiss not available, falling back to brute-force retrieval")
            return self._retrieve_brute(query, top_k)

    # ------------------------------------------------------------------ I/O

    def save(self, path: Path) -> None:
        """Persist buffer to a .npz archive.

        Array layout (all parallel arrays of length N = len(buffer)):
          embeddings      (N, D)
          action_chunks   (N, chunk_len, action_dim)
          footprints      (N,)
          progresses      (N,)
          task_descs      (N,) object array of str
          source_episodes (N,) object array of str
          steps           (N,) int32
          max_size        scalar
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self._entries) == 0:
            np.savez(path, embeddings=np.empty((0,)), max_size=np.array(self.max_size))
            return

        np.savez(
            path,
            embeddings      = np.stack([e.embedding     for e in self._entries]),
            action_chunks   = np.stack([e.action_chunk  for e in self._entries]),
            footprints      = np.array([e.footprint      for e in self._entries], dtype=np.float32),
            progresses      = np.array([e.progress       for e in self._entries], dtype=np.float32),
            task_descs      = np.array([e.task_description for e in self._entries], dtype=object),
            source_episodes = np.array([e.source_episode  for e in self._entries], dtype=object),
            steps           = np.array([e.step             for e in self._entries], dtype=np.int32),
            max_size        = np.array(self.max_size),
        )
        logger.info("SAB saved: %d entries -> %s", len(self._entries), path)

    def load(self, path: Path) -> None:
        """Load buffer from a .npz archive (replaces current contents)."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        self.max_size = int(data["max_size"])
        self._entries = []

        if "action_chunks" not in data or data["embeddings"].ndim == 1:
            return  # empty buffer file

        embeddings      = data["embeddings"]
        action_chunks   = data["action_chunks"]
        footprints      = data["footprints"]
        progresses      = data["progresses"]
        task_descs      = data["task_descs"]
        source_episodes = data["source_episodes"]
        steps           = data["steps"]

        for i in range(len(embeddings)):
            self._entries.append(BufferEntry(
                embedding       = embeddings[i],
                action_chunk    = action_chunks[i],
                footprint       = float(footprints[i]),
                progress        = float(progresses[i]),
                task_description= str(task_descs[i]),
                source_episode  = str(source_episodes[i]),
                step            = int(steps[i]),
            ))
        logger.info("SAB loaded: %d entries from %s", len(self._entries), path)

    @classmethod
    def from_file(cls, path: Path, max_size: int = 50_000) -> "SafetyAlternativeBuffer":
        """Convenience factory: create and load in one call."""
        buf = cls(max_size=max_size)
        buf.load(path)
        return buf

    # -------------------------------------------------------- geometry summary

    @staticmethod
    def make_embedding_key(
        vla_embedding: np.ndarray,               # (D,)
        geo_summary:   Optional[np.ndarray] = None,  # (K_geo,) or None
    ) -> np.ndarray:
        """Concatenate VLA embedding with optional geometric summary.

        The geometric summary is a low-dimensional vector encoding the
        positions of the K most relevant scene objects (e.g. the 3D
        centroid of the target + protected objects concatenated, shape
        3*N_objects).  If None, returns the VLA embedding unchanged.
        """
        if geo_summary is None or geo_summary.size == 0:
            return vla_embedding.astype(np.float32)
        key = np.concatenate([vla_embedding, geo_summary]).astype(np.float32)
        return key
