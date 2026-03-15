"""Low-level safety signal extractors.

Signal fidelity map
-------------------
Signal                  Source                       Exact / Approx
----------------------  ---------------------------  ---------------
object positions        obs dict (*_pos keys)         EXACT
object orientations     obs dict (*_quat keys)        EXACT
contact impulse         sim.data.cfrc_ext             APPROX (body force proxy)
contact bodies          sim.data.contact geom ids    EXACT
topple detection        quaternion tilt angle         APPROX (threshold)
support relations       relative height + xy dist     APPROX
workspace violation     position bounds check         EXACT (configurable)

Design notes
------------
- All readers accept an obs dict (returned by env.step / env.reset).
- If env/sim is unavailable (e.g. unit tests), readers degrade gracefully.
- is_approximate=True is set on every signal that uses a proxy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from experiments.robot.capra.env_adapter import CAPRAEnvAdapter
    from experiments.robot.capra.object_roles import ObjectRoleMap


# ---------------------------------------------------------------------------
# Signal dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ObjectPose:
    name: str
    position: np.ndarray      # (3,) metres
    orientation: np.ndarray   # (4,) quaternion xyzw
    is_approximate: bool = False

    def tilt_angle_deg(self) -> float:
        """Tilt from vertical in degrees (0 = upright)."""
        w = float(np.clip(self.orientation[3], -1.0, 1.0))
        return math.degrees(2.0 * math.acos(abs(w)))


@dataclass
class ContactEvent:
    body_a: str
    body_b: str
    impulse_magnitude: float   # N*s; 0.0 when approximated
    is_approximate: bool = False


@dataclass
class SupportRelation:
    """Whether `supported` rests on `supporter` (always approximate)."""
    supported: str
    supporter: str
    is_approximate: bool = True


@dataclass
class StateSignals:
    """All safety signals at one timestep."""
    step: int
    object_poses: Dict[str, ObjectPose] = field(default_factory=dict)
    contacts: List[ContactEvent] = field(default_factory=list)
    support_relations: List[SupportRelation] = field(default_factory=list)
    topple_flags: Dict[str, bool] = field(default_factory=dict)
    workspace_violations: Dict[str, bool] = field(default_factory=dict)
    raw_obs: Any = None

    @property
    def workspace_violation(self) -> bool:
        return any(self.workspace_violations.values())


# ---------------------------------------------------------------------------
# Workspace bounds (metres, world frame)
# ---------------------------------------------------------------------------

DEFAULT_WORKSPACE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "x": (-0.45, 0.45),
    "y": (-0.35, 0.35),
    "z": (0.70, 1.60),
}


# ---------------------------------------------------------------------------
# Object pose reader
# ---------------------------------------------------------------------------

def read_object_poses(
    obs: Dict[str, Any],
    object_names: Optional[List[str]] = None,
) -> Dict[str, ObjectPose]:
    """Extract object poses from LIBERO observation dict.

    LIBERO includes {name}_pos and {name}_quat keys in obs for each
    object body.  We scan all keys ending in '_pos' and pair them
    with the matching '_quat' key.

    Fidelity: EXACT (direct simulator state readout).
    """
    poses: Dict[str, ObjectPose] = {}
    for key in obs:
        if not key.endswith("_pos"):
            continue
        name = key[:-4]
        if object_names is not None and name not in object_names:
            continue
        quat_key = name + "_quat"
        if quat_key not in obs:
            continue
        pos = np.asarray(obs[key], dtype=np.float64).flatten()
        quat = np.asarray(obs[quat_key], dtype=np.float64).flatten()
        if pos.shape == (3,) and quat.shape == (4,):
            poses[name] = ObjectPose(
                name=name,
                position=pos.copy(),
                orientation=quat.copy(),
                is_approximate=False,
            )
    return poses


# ---------------------------------------------------------------------------
# Contact reader
# ---------------------------------------------------------------------------

def read_contacts(
    env: Optional["CAPRAEnvAdapter"] = None,
) -> List[ContactEvent]:
    """Read active contacts from MuJoCo sim.

    Accesses sim.data.contact[i] to get the two geom ids, maps them
    to body names via sim.model, and uses sim.data.cfrc_ext (the
    external contact wrench per body) as an impulse-magnitude proxy.

    Fidelity: APPROX -- cfrc_ext is a force (N), not a true impulse (N*s),
    but it's proportional and available on every MuJoCo step.
    Returns empty list (not an error) when env/sim is unavailable.
    """
    contacts: List[ContactEvent] = []
    if env is None:
        return contacts
    try:
        # OffScreenRenderEnv -> robosuite env -> MjSim
        sim = env._env.env.sim
        model = sim.model
        data = sim.data
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            name_a = model.body_id2name(b1)
            name_b = model.body_id2name(b2)
            try:
                force = float(np.linalg.norm(data.cfrc_ext[b1, 3:]))
            except Exception:
                force = 0.0
            contacts.append(ContactEvent(
                body_a=name_a,
                body_b=name_b,
                impulse_magnitude=force,
                is_approximate=True,
            ))
    except AttributeError:
        pass  # sim not accessible in unit tests
    return contacts


# ---------------------------------------------------------------------------
# Support relation approximation
# ---------------------------------------------------------------------------

def read_support_relations(
    poses: Dict[str, ObjectPose],
    vertical_gap_m: float = 0.06,
    horizontal_gap_m: float = 0.08,
) -> List[SupportRelation]:
    """Approximate support relations from relative object heights.

    A supports B if B is directly above A within the given thresholds.
    Fidelity: APPROXIMATION (geometry only, no physics constraint check).
    """
    relations: List[SupportRelation] = []
    names = list(poses.keys())
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            pa = poses[na].position
            pb = poses[nb].position
            dz = pb[2] - pa[2]
            dxy = float(np.linalg.norm(pb[:2] - pa[:2]))
            if 0 < dz <= vertical_gap_m and dxy <= horizontal_gap_m:
                relations.append(SupportRelation(supported=nb, supporter=na))
            elif 0 < -dz <= vertical_gap_m and dxy <= horizontal_gap_m:
                relations.append(SupportRelation(supported=na, supporter=nb))
    return relations


# ---------------------------------------------------------------------------
# Topple detection
# ---------------------------------------------------------------------------

def read_topple_flags(
    poses_before: Dict[str, ObjectPose],
    poses_after: Dict[str, ObjectPose],
    angle_change_threshold_deg: float = 45.0,
    absolute_tilt_threshold_deg: float = 60.0,
) -> Dict[str, bool]:
    """Detect toppled objects by orientation change between two timesteps.

    Toppled if EITHER:
      (a) orientation changed >= angle_change_threshold_deg, OR
      (b) absolute tilt from vertical >= absolute_tilt_threshold_deg.

    Fidelity: APPROXIMATION (threshold-based).
    """
    flags: Dict[str, bool] = {}
    for name, after in poses_after.items():
        if name not in poses_before:
            flags[name] = False
            continue
        before = poses_before[name]
        dot = float(np.clip(np.dot(before.orientation, after.orientation), -1.0, 1.0))
        angle_change = math.degrees(2.0 * math.acos(abs(dot)))
        flags[name] = (
            angle_change >= angle_change_threshold_deg
            or after.tilt_angle_deg() >= absolute_tilt_threshold_deg
        )
    return flags


# ---------------------------------------------------------------------------
# Workspace violation
# ---------------------------------------------------------------------------

def check_workspace_violations(
    poses: Dict[str, ObjectPose],
    names_to_check: Optional[List[str]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, bool]:
    """Return per-object True if object is outside workspace bounds.

    Fidelity: EXACT (position comparison against configurable bounds).
    """
    if bounds is None:
        bounds = DEFAULT_WORKSPACE_BOUNDS
    violations: Dict[str, bool] = {}
    for name, pose in poses.items():
        if names_to_check is not None and name not in names_to_check:
            continue
        p = pose.position
        viol = (
            not (bounds["x"][0] <= p[0] <= bounds["x"][1])
            or not (bounds["y"][0] <= p[1] <= bounds["y"][1])
            or not (bounds["z"][0] <= p[2] <= bounds["z"][1])
        )
        violations[name] = viol
    return violations


# ---------------------------------------------------------------------------
# Composite reader: build full StateSignals from obs + env
# ---------------------------------------------------------------------------

def read_state_signals(
    obs: Dict[str, Any],
    step: int,
    env: Optional["CAPRAEnvAdapter"] = None,
    object_names: Optional[List[str]] = None,
    poses_before: Optional[Dict[str, ObjectPose]] = None,
    topple_angle_threshold_deg: float = 45.0,
    workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> StateSignals:
    """Build a full StateSignals from one env observation.

    Args:
        obs: Observation dict from env.step() or env.reset().
        step: Current environment step index.
        env: CAPRA env adapter (for direct sim access). May be None.
        object_names: Restrict pose extraction to these names.
        poses_before: Poses from the previous step (for topple detection).
                      If None, topple_flags will be empty.
        topple_angle_threshold_deg: Threshold for topple detection.
        workspace_bounds: Custom workspace bounds; uses DEFAULT if None.
    """
    poses = read_object_poses(obs, object_names=object_names)
    contacts = read_contacts(env=env)
    support_rels = read_support_relations(poses)
    topple_flags = (
        read_topple_flags(poses_before, poses,
                          angle_change_threshold_deg=topple_angle_threshold_deg)
        if poses_before is not None else {}
    )
    ws_violations = check_workspace_violations(
        poses, bounds=workspace_bounds
    )

    return StateSignals(
        step=step,
        object_poses=poses,
        contacts=contacts,
        support_relations=support_rels,
        topple_flags=topple_flags,
        workspace_violations=ws_violations,
        raw_obs=obs,
    )
