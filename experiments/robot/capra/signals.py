"""Low-level safety signal extractors for footprint computation.

Three components of F_t(a) are computed here:
  1. non_target_displacement  -- Σ ||Δpos_i|| for non-target objects
  2. contact_impulse          -- total impulse on protected objects
  3. irreversible_events      -- topple / support-break / workspace exit

Each component is object-wise and task-conditioned: only objects that
should NOT be moved at the current task stage are penalised.

Phase 1: dataclass + stubs.
Phase 2: wire to state_api.py readers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.state_api import StateSignals
    from experiments.robot.capra.object_roles import ObjectRoleMap


@dataclass
class FootprintComponents:
    """Raw components before weighting."""
    non_target_displacement: float = 0.0   # metres, sum over non-target objects
    contact_impulse: float = 0.0           # N·s, sum over protected objects
    irreversible_count: float = 0.0        # number of irreversible events
    topple_count: int = 0
    support_break_count: int = 0
    workspace_violation: bool = False
    is_approximate: bool = False


def compute_non_target_displacement(
    signals_before: "StateSignals",
    signals_after: "StateSignals",
    role_map: "ObjectRoleMap",
) -> float:
    """Sum of positional displacement of objects with roles NON_TARGET or PROTECTED."""
    raise NotImplementedError("Phase 2.")


def compute_contact_impulse(
    signals_after: "StateSignals",
    role_map: "ObjectRoleMap",
) -> float:
    """Total contact impulse on PROTECTED objects during the rollout."""
    raise NotImplementedError(
        "Phase 2: approximate as 0.0 if impulse not available from backend."
    )


def compute_irreversible_events(
    signals_before: "StateSignals",
    signals_after: "StateSignals",
    role_map: "ObjectRoleMap",
) -> FootprintComponents:
    """Detect topple, support-break, and workspace-violation events."""
    raise NotImplementedError("Phase 2.")


def aggregate_footprint_components(
    signals_before: "StateSignals",
    signals_after: "StateSignals",
    role_map: "ObjectRoleMap",
) -> FootprintComponents:
    """Compute all three footprint components from a before/after signal pair."""
    raise NotImplementedError("Phase 2.")
