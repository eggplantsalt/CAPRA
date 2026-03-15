"""Footprint F_t(a): weighted combination of safety signal components.

F_t(a) = alpha_d * displacement + alpha_i * impulse + alpha_r * irreversible

All weights are in CAPRAConfig and must not be hard-coded here.

Phase 1: interface + stub.
Phase 2: wire to signals.py and aggregate.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.robot.capra.capra_config import CAPRAConfig
    from experiments.robot.capra.signals import FootprintComponents
    from experiments.robot.capra.object_roles import ObjectRoleMap
    from experiments.robot.capra.state_api import StateSignals
    from experiments.robot.capra.env_adapter import CAPRAEnvAdapter
    from experiments.robot.capra.snapshot import Snapshot

import numpy as np


def compute_footprint(
    components: "FootprintComponents",
    cfg: "CAPRAConfig",
) -> float:
    """Combine footprint components into a scalar F_t(a).

    F_t(a) = alpha_d * displacement
           + alpha_i * impulse
           + alpha_r * irreversible_count
    """
    return (
        cfg.alpha_d * components.non_target_displacement
        + cfg.alpha_i * components.contact_impulse
        + cfg.alpha_r * components.irreversible_count
    )


def evaluate_footprint_for_action(
    env: "CAPRAEnvAdapter",
    snap: "Snapshot",
    action_chunk: np.ndarray,
    role_map: "ObjectRoleMap",
    cfg: "CAPRAConfig",
) -> float:
    """Run a short counterfactual rollout from snap and return F_t(action_chunk).

    Phase 2: restore snapshot, execute chunk, read signals, call compute_footprint.
    """
    raise NotImplementedError("Phase 2: requires snapshot restore + state_api.")
