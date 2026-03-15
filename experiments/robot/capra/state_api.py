"""Environment state signal API.

Reads object poses, contact impulses, support relationships, and
irreversibility flags from a LIBERO simulation step.

All signals that cannot be read exactly are flagged as APPROXIMATED.
Approximation logic will live in dedicated helper functions so it can
be swapped independently of the training loop.

Phase 1: dataclass definitions + stubs.
Phase 2: fill backend read logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.env_adapter import CAPRAEnvAdapter


# ----------------------------------------------------------------- signal types

@dataclass
class ObjectPose:
    name: str
    position: np.ndarray       # shape (3,)
    orientation: np.ndarray    # quaternion, shape (4,)
    is_approximate: bool = False


@dataclass
class ContactEvent:
    body_a: str
    body_b: str
    impulse_magnitude: float   # N·s; 0.0 if not directly available
    is_approximate: bool = False


@dataclass
class SupportRelation:
    """Whether object `supported` rests on object `supporter`."""
    supported: str
    supporter: str
    is_approximate: bool = True  # always approximated unless physics engine exposes it


@dataclass
class StateSignals:
    """All signals read at one timestep."""
    step: int
    object_poses: Dict[str, ObjectPose] = field(default_factory=dict)
    contacts: List[ContactEvent] = field(default_factory=list)
    support_relations: List[SupportRelation] = field(default_factory=list)
    topple_flags: Dict[str, bool] = field(default_factory=dict)   # object_name -> toppled?
    workspace_violation: bool = False
    raw_obs: Any = None  # full env obs dict for downstream consumers


# ----------------------------------------------------------------- reader stubs

def read_state_signals(env: "CAPRAEnvAdapter", obs: Dict, step: int) -> StateSignals:
    """Read all CAPRA state signals from one environment observation.

    Phase 2: implement each sub-reader against the MuJoCo/LIBERO backend.
    """
    raise NotImplementedError("Phase 2: implement per-signal readers.")


def read_object_poses(env: "CAPRAEnvAdapter", obs: Dict) -> Dict[str, ObjectPose]:
    """Extract object poses from observation dict or sim state."""
    raise NotImplementedError("Phase 2.")


def read_contacts(env: "CAPRAEnvAdapter") -> List[ContactEvent]:
    """Extract contact/impulse data from physics engine."""
    raise NotImplementedError(
        "Phase 2: approximate as zero impulse if backend does not expose contacts."
    )


def read_support_relations(env: "CAPRAEnvAdapter", poses: Dict[str, ObjectPose]) -> List[SupportRelation]:
    """Approximate support relations from relative object heights and overlap."""
    raise NotImplementedError("Phase 2.")


def read_topple_flags(poses_before: Dict[str, ObjectPose],
                     poses_after: Dict[str, ObjectPose],
                     angle_threshold_deg: float = 45.0) -> Dict[str, bool]:
    """Detect toppling events by orientation change between two timesteps."""
    raise NotImplementedError("Phase 2.")
