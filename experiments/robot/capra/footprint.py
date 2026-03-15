"""Footprint F_t(a): object-wise, decomposable, task-conditioned.

Decomposition
-------------
F_t(a) = alpha_d * D_t(a)   +   alpha_i * I_t(a)   +   alpha_r * R_t(a)

  D_t(a)  non-target displacement cost
          = sum_{o in penalised} weight(o) * ||pos_after(o) - pos_before(o)||
          Exact signal: object positions from obs dict.

  I_t(a)  contact impulse cost
          = sum_{o in protected} total_impulse_on(o)
          Approx signal: sim.data.cfrc_ext (body contact force proxy).

  R_t(a)  irreversible event cost
          = topple_count * w_topple
          + support_break_count * w_support_break
          + workspace_violation_count * w_workspace
          Approx signal: threshold-based from quaternion change + position bounds.

Task-conditioning
-----------------
Only objects with ObjectRole.PROTECTED or NON_TARGET are charged.
TARGET displacement is not penalised (moving the target is the goal).
IRRELEVANT objects (furniture, walls) are ignored entirely.
Weights are per-object via ObjectRoleMap.get_weight(name).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.capra_config import CAPRAConfig
    from experiments.robot.capra.object_roles import ObjectRoleMap
    from experiments.robot.capra.signals import StateSignals


# ---------------------------------------------------------------------------
# FootprintComponents: raw, un-weighted values before alpha scaling
# ---------------------------------------------------------------------------

@dataclass
class FootprintComponents:
    """Object-wise raw components of F_t(a).

    All values are BEFORE multiplication by alpha_d / alpha_i / alpha_r.
    The `per_object_displacement` dict lets callers inspect which objects
    contributed most to the displacement cost.
    """
    # --- Component 1: non-target displacement ---
    non_target_displacement: float = 0.0   # metres, weighted sum
    per_object_displacement: Dict[str, float] = field(default_factory=dict)

    # --- Component 2: contact impulse ---
    contact_impulse: float = 0.0           # N (force proxy), on protected objects
    per_object_impulse: Dict[str, float] = field(default_factory=dict)

    # --- Component 3: irreversible events ---
    irreversible_count: float = 0.0        # weighted event count
    topple_count: int = 0
    support_break_count: int = 0
    workspace_violation_count: int = 0

    # Fidelity bookkeeping
    displacement_is_approximate: bool = False
    impulse_is_approximate: bool = True    # always approx (force proxy)
    irreversible_is_approximate: bool = True

    def log_str(self) -> str:
        """One-line summary for logging."""
        top3_disp = sorted(
            self.per_object_displacement.items(), key=lambda x: -x[1]
        )[:3]
        top3_str = " ".join(f"{n}:{v:.3f}" for n, v in top3_disp)
        return (
            f"FootprintComponents("
            f"disp={self.non_target_displacement:.4f}m "
            f"impulse={self.contact_impulse:.4f}N "
            f"irrev={self.irreversible_count:.1f} "
            f"[topple={self.topple_count} supp_brk={self.support_break_count} "
            f"ws_viol={self.workspace_violation_count}] "
            f"top_disp=[{top3_str}])"
        )


# ---------------------------------------------------------------------------
# Component 1: non-target displacement
# ---------------------------------------------------------------------------

def compute_non_target_displacement(
    signals_before: "StateSignals",
    signals_after: "StateSignals",
    role_map: "ObjectRoleMap",
) -> tuple[float, Dict[str, float]]:
    """Weighted sum of positional displacement for penalised objects.

    Only PROTECTED and NON_TARGET objects are charged.
    TARGET objects are excluded (their movement is desired).
    Per-object weight comes from ObjectRoleMap.get_weight(name).

    Returns:
        total_displacement: float
        per_object: dict of name -> weighted displacement
    """
    total = 0.0
    per_obj: Dict[str, float] = {}

    for name in role_map.penalised_objects():
        if name not in signals_before.object_poses:
            continue
        if name not in signals_after.object_poses:
            continue
        pos_before = signals_before.object_poses[name].position
        pos_after = signals_after.object_poses[name].position
        dist = float(np.linalg.norm(pos_after - pos_before))
        weight = role_map.get_weight(name)
        weighted = weight * dist
        per_obj[name] = weighted
        total += weighted

    return total, per_obj


# ---------------------------------------------------------------------------
# Component 2: contact impulse on protected objects
# ---------------------------------------------------------------------------

def compute_contact_impulse(
    signals_after: "StateSignals",
    role_map: "ObjectRoleMap",
) -> tuple[float, Dict[str, float]]:
    """Sum of contact force magnitudes on PROTECTED objects.

    Uses signals_after.contacts (ContactEvent list).  Only contacts
    where body_a OR body_b is a PROTECTED object are counted.

    Fidelity: APPROX (cfrc_ext force proxy, not true impulse).

    Returns:
        total_impulse: float
        per_object: dict of name -> impulse attributed to that object
    """
    from experiments.robot.capra.object_roles import ObjectRole

    total = 0.0
    per_obj: Dict[str, float] = {}

    for event in signals_after.contacts:
        for body in (event.body_a, event.body_b):
            if role_map.get_role(body) == ObjectRole.PROTECTED:
                weight = role_map.get_weight(body)
                contrib = weight * event.impulse_magnitude
                per_obj[body] = per_obj.get(body, 0.0) + contrib
                total += contrib
                break  # count each contact event once

    return total, per_obj


# ---------------------------------------------------------------------------
# Component 3: irreversible events
# ---------------------------------------------------------------------------

# Relative weights for different irreversible event types
_W_TOPPLE = 1.0
_W_SUPPORT_BREAK = 1.5
_W_WORKSPACE = 2.0


def compute_irreversible_events(
    signals_before: "StateSignals",
    signals_after: "StateSignals",
    role_map: "ObjectRoleMap",
) -> tuple[float, int, int, int]:
    """Count weighted irreversible events affecting penalised objects.

    Three sub-events:
      topple        -- object tipped over (from topple_flags).
      support_break -- a support relation present before is gone after
                       AND the former supported object has also moved.
      workspace_viol-- object left workspace bounds.

    Only PROTECTED and NON_TARGET objects are counted.

    Returns:
        weighted_count: float  (weighted sum using _W_* constants)
        topple_count: int
        support_break_count: int
        workspace_violation_count: int
    """
    penalised = set(role_map.penalised_objects())
    weighted = 0.0
    n_topple = 0
    n_supp_brk = 0
    n_ws = 0

    # -- Topples --
    for name, toppled in signals_after.topple_flags.items():
        if toppled and name in penalised:
            weighted += _W_TOPPLE * role_map.get_weight(name)
            n_topple += 1

    # -- Support breaks --
    # A support break occurs when a relation (supporter -> supported) present
    # in signals_before disappears in signals_after AND the supported object
    # has meaningfully moved.
    before_pairs = {
        (r.supporter, r.supported) for r in signals_before.support_relations
    }
    after_pairs = {
        (r.supporter, r.supported) for r in signals_after.support_relations
    }
    broken = before_pairs - after_pairs
    for supporter, supported in broken:
        if supported not in penalised:
            continue
        # Only flag if the supported object actually moved
        if supported in signals_before.object_poses and supported in signals_after.object_poses:
            disp = float(np.linalg.norm(
                signals_after.object_poses[supported].position
                - signals_before.object_poses[supported].position
            ))
            if disp > 0.02:  # 2 cm movement confirms the break
                weighted += _W_SUPPORT_BREAK * role_map.get_weight(supported)
                n_supp_brk += 1

    # -- Workspace violations --
    for name, violated in signals_after.workspace_violations.items():
        if violated and name in penalised:
            # Only charge if NOT already violated before (new event)
            was_violated = signals_before.workspace_violations.get(name, False)
            if not was_violated:
                weighted += _W_WORKSPACE * role_map.get_weight(name)
                n_ws += 1

    return weighted, n_topple, n_supp_brk, n_ws


# ---------------------------------------------------------------------------
# Aggregate all three components
# ---------------------------------------------------------------------------

def aggregate_footprint_components(
    signals_before: "StateSignals",
    signals_after: "StateSignals",
    role_map: "ObjectRoleMap",
) -> FootprintComponents:
    """Compute all three footprint components from a before/after signal pair."""
    disp_total, per_disp = compute_non_target_displacement(
        signals_before, signals_after, role_map
    )
    impulse_total, per_impulse = compute_contact_impulse(
        signals_after, role_map
    )
    irrev_weighted, n_topple, n_supp, n_ws = compute_irreversible_events(
        signals_before, signals_after, role_map
    )

    return FootprintComponents(
        non_target_displacement=disp_total,
        per_object_displacement=per_disp,
        contact_impulse=impulse_total,
        per_object_impulse=per_impulse,
        irreversible_count=irrev_weighted,
        topple_count=n_topple,
        support_break_count=n_supp,
        workspace_violation_count=n_ws,
        displacement_is_approximate=False,
        impulse_is_approximate=True,
        irreversible_is_approximate=True,
    )


# ---------------------------------------------------------------------------
# Scalar footprint
# ---------------------------------------------------------------------------

def compute_footprint(
    components: "FootprintComponents",
    cfg: "CAPRAConfig",
) -> float:
    """Combine components into F_t(a) scalar.

    F_t(a) = alpha_d * D_t  +  alpha_i * I_t  +  alpha_r * R_t
    """
    return (
        cfg.alpha_d * components.non_target_displacement
        + cfg.alpha_i * components.contact_impulse
        + cfg.alpha_r * components.irreversible_count
    )
