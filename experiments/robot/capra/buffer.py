"""Safety Alternative Buffer (SAB).

The SAB is a training-time-only offline retrieval store. It stores
(embedding, action_chunk) pairs where the action_chunk was identified
as a low-footprint, task-equivalent alternative to a higher-risk
action taken by the base policy.

CRITICAL: The buffer MUST NOT enter the test-time model structure.
It is used exclusively during offline mining (run_capra_mining.py)
and CAPRA dataset building (build_capra_dataset.py).

Retrieval representation: frozen pre-action visual-language embedding
concatenated with a low-dimensional geometric summary (object poses).

Phase 1: schema + interface stubs.
Phase 2: implement insert / retrieve / persist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BufferEntry:
    """One record in the Safety Alternative Buffer."""
    embedding: np.ndarray          # retrieval key  (D,)
    action_chunk: np.ndarray       # safe alternative  (chunk_len, action_dim)
    footprint: float               # F_t of this action
    progress: float                # P_t of this action
    task_description: str = ""
    source_episode: str = ""
    step: int = 0


class SafetyAlternativeBuffer:
    """In-memory store with numpy-based nearest-neighbour retrieval.

    This is intentionally simple. If retrieval becomes a bottleneck
    a FAISS index can be dropped in without changing the interface.
    """

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self._entries: List[BufferEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def insert(self, entry: BufferEntry) -> None:
        """Add an entry; evict oldest if at capacity."""
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)
        self._entries.append(entry)

    def retrieve(
        self,
        query_embedding: np.ndarray,  # (D,)
        top_k: int = 4,
    ) -> List[BufferEntry]:
        """Return the top_k entries closest to query_embedding (L2).

        Phase 2: replace linear scan with FAISS if needed.
        """
        if len(self._entries) == 0:
            return []
        embeddings = np.stack([e.embedding for e in self._entries])  # (N, D)
        dists = np.linalg.norm(embeddings - query_embedding[None], axis=1)
        top_idx = np.argsort(dists)[:top_k]
        return [self._entries[i] for i in top_idx]

    def save(self, path: Path) -> None:
        """Persist buffer to disk as a numpy archive."""
        raise NotImplementedError("Phase 2.")

    def load(self, path: Path) -> None:
        """Load buffer from a numpy archive."""
        raise NotImplementedError("Phase 2.")
