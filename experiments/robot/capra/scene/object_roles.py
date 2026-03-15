"""Object role taxonomy for CAPRA footprint computation.

Every object in the scene is assigned one of four roles at the start
of each task. Roles drive footprint weighting; raw object names are
never hard-coded in the training loop.

Roles
-----
TARGET      the object the robot is currently instructed to manipulate.
            Displacement of this object is DESIRED, not penalised.
PROTECTED   safety-critical objects that must not be displaced.
            Carry the highest footprint penalty.
NON_TARGET  other movable objects (clutter, neighbouring items).
            Carry a smaller penalty.
IRRELEVANT fixed furniture, walls, floor -- ignored in footprint.

Role assignment
---------------
The primary assignment path is `assign_roles_from_task_description`,
which uses simple keyword heuristics over the task language string.
This is an approximation but is entirely self-contained and needs no
external metadata.  The exact assignment is clearly flagged in every
ObjectRoleMap via the `assignment_method` field so analysis scripts
can distinguish exact vs. approximate assignments.

For future work, `assign_roles_from_bddl` accepts raw BDDL text and
parses the target object more precisely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import re


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------

class ObjectRole(Enum):
    TARGET = auto()
    PROTECTED = auto()
    NON_TARGET = auto()
    IRRELEVANT = auto()


# Default per-role footprint cost multipliers (can be overridden in CAPRAConfig)
ROLE_DEFAULT_WEIGHT: Dict[ObjectRole, float] = {
    ObjectRole.TARGET:     0.0,   # moving the target is desired
    ObjectRole.PROTECTED:  2.0,   # highest penalty
    ObjectRole.NON_TARGET: 1.0,   # standard penalty
    ObjectRole.IRRELEVANT: 0.0,   # ignore
}


# ---------------------------------------------------------------------------
# ObjectRoleMap
# ---------------------------------------------------------------------------

@dataclass
class ObjectRoleMap:
    """Mapping from object name to role for one task episode.

    Object names should match the keys returned by the environment's
    object-pose API (typically the MuJoCo body name, lower-cased).
    """
    target: List[str] = field(default_factory=list)
    protected: List[str] = field(default_factory=list)
    non_target: List[str] = field(default_factory=list)
    irrelevant: List[str] = field(default_factory=list)

    # How this map was built -- used for logging / analysis
    assignment_method: str = "unknown"  # "heuristic" | "bddl" | "manual"

    # Per-object weight overrides: name -> multiplier
    # If absent the role default from ROLE_DEFAULT_WEIGHT is used.
    weight_overrides: Dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- lookups

    def get_role(self, name: str) -> ObjectRole:
        if name in self.target:
            return ObjectRole.TARGET
        if name in self.protected:
            return ObjectRole.PROTECTED
        if name in self.non_target:
            return ObjectRole.NON_TARGET
        return ObjectRole.IRRELEVANT

    def get_weight(self, name: str) -> float:
        if name in self.weight_overrides:
            return self.weight_overrides[name]
        return ROLE_DEFAULT_WEIGHT[self.get_role(name)]

    def penalised_objects(self) -> List[str]:
        """Objects with non-zero footprint weight (PROTECTED + NON_TARGET)."""
        return self.protected + self.non_target

    def all_names(self) -> List[str]:
        return self.target + self.protected + self.non_target + self.irrelevant

    def summary(self) -> str:
        return (
            f"ObjectRoleMap(method={self.assignment_method} "
            f"target={self.target} protected={self.protected} "
            f"non_target={self.non_target[:3]}{'...' if len(self.non_target) > 3 else ''})"
        )


# ---------------------------------------------------------------------------
# Heuristic assignment from task description (primary approximation path)
# ---------------------------------------------------------------------------

# Fixed furniture / structure keywords -> IRRELEVANT
_IRRELEVANT_KEYWORDS = (
    "table", "desk", "shelf", "cabinet", "drawer", "floor", "wall",
    "base", "robot", "gripper", "robot0", "pedestal", "mount",
)

# High-risk support-structure keywords -> PROTECTED (when not the target)
_PROTECTED_KEYWORDS = (
    "stack", "tower", "support", "holder", "rack", "tray",
)


def _tokenise(text: str) -> List[str]:
    """Lower-case, split on non-alphanumeric chars."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _extract_target_from_description(task_description: str) -> Optional[str]:
    """Heuristically extract the manipulation target object name.

    Looks for patterns like "pick up the OBJECT", "place the OBJECT",
    "push the OBJECT", "grasp the OBJECT", "move the OBJECT".

    Returns the first candidate token, or None if no match.
    """
    # Patterns: verb + (optional 'the'/'a') + NOUN
    patterns = [
        r"(?:pick\s+up|grasp|grab|lift|take)\s+(?:the\s+|a\s+)?([a-z0-9_]+)",
        r"(?:place|put|move|push|slide|stack|insert|open|close)\s+(?:the\s+|a\s+)?([a-z0-9_]+)",
        r"(?:turn\s+on|turn\s+off)\s+(?:the\s+|a\s+)?([a-z0-9_]+)",
    ]
    lower = task_description.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            candidate = m.group(1)
            # Skip if it's clearly a preposition / stop word
            if candidate not in ("it", "them", "object", "item", "thing"):
                return candidate
    return None


def assign_roles_from_task_description(
    task_description: str,
    object_names: List[str],
    extra_protected: Optional[List[str]] = None,
) -> ObjectRoleMap:
    """Assign roles using keyword heuristics over the task description.

    This is an APPROXIMATION -- flagged in role_map.assignment_method.
    It is the default path when no BDDL metadata is available.

    Logic:
      1. Extract target object via verb-noun pattern in task_description.
      2. Objects matching _IRRELEVANT_KEYWORDS -> IRRELEVANT.
      3. Objects matching _PROTECTED_KEYWORDS (and not target) -> PROTECTED.
      4. Any object in extra_protected -> PROTECTED.
      5. Remaining movable objects -> NON_TARGET.

    Args:
        task_description: Natural language task string.
        object_names: All object body names from the scene.
        extra_protected: Additional names to force into PROTECTED.
    """
    target_hint = _extract_target_from_description(task_description)
    extra_protected = extra_protected or []

    target: List[str] = []
    protected: List[str] = []
    non_target: List[str] = []
    irrelevant: List[str] = []

    for name in object_names:
        lower = name.lower()

        # Step 1: check irrelevant keywords
        if any(kw in lower for kw in _IRRELEVANT_KEYWORDS):
            irrelevant.append(name)
            continue

        # Step 2: match against extracted target hint
        if target_hint and target_hint in lower:
            target.append(name)
            continue

        # Step 3: forced protected list
        if name in extra_protected:
            protected.append(name)
            continue

        # Step 4: protected by keyword
        if any(kw in lower for kw in _PROTECTED_KEYWORDS):
            protected.append(name)
            continue

        # Step 5: everything else is non-target movable
        non_target.append(name)

    # If no target was found, treat the first non-target as target (fallback)
    if not target and non_target:
        target.append(non_target.pop(0))

    return ObjectRoleMap(
        target=target,
        protected=protected,
        non_target=non_target,
        irrelevant=irrelevant,
        assignment_method="heuristic",
    )


def assign_roles_from_bddl(
    bddl_text: str,
    object_names: List[str],
    extra_protected: Optional[List[str]] = None,
) -> ObjectRoleMap:
    """Assign roles by parsing BDDL task description.

    BDDL typically contains a `(:goal ...)` block that mentions the
    target object in predicates like `(On ?obj ?surface)`.
    This gives a more precise target identification than heuristics.

    Approximation level: SEMI-EXACT for target; heuristic for others.
    """
    extra_protected = extra_protected or []

    # Extract object mentioned in goal predicates
    goal_match = re.search(r"\(:goal.*?\)", bddl_text, re.DOTALL | re.IGNORECASE)
    target_candidates: List[str] = []
    if goal_match:
        # Find all ?variable names in goal
        vars_in_goal = re.findall(r"\?([a-z0-9_]+)", goal_match.group(0).lower())
        # Cross-reference with object_names
        for obj in object_names:
            if any(v in obj.lower() for v in vars_in_goal):
                target_candidates.append(obj)

    # Fall back to heuristic description extraction
    target_hint = target_candidates[0] if target_candidates else None

    target: List[str] = []
    protected: List[str] = []
    non_target: List[str] = []
    irrelevant: List[str] = []

    for name in object_names:
        lower = name.lower()
        if any(kw in lower for kw in _IRRELEVANT_KEYWORDS):
            irrelevant.append(name)
        elif target_hint and target_hint.lower() in lower:
            target.append(name)
        elif name in extra_protected or any(kw in lower for kw in _PROTECTED_KEYWORDS):
            protected.append(name)
        else:
            non_target.append(name)

    if not target and non_target:
        target.append(non_target.pop(0))

    return ObjectRoleMap(
        target=target,
        protected=protected,
        non_target=non_target,
        irrelevant=irrelevant,
        assignment_method="bddl",
    )


def assign_roles_manual(
    target: List[str],
    protected: List[str],
    non_target: List[str],
    irrelevant: Optional[List[str]] = None,
    weight_overrides: Optional[Dict[str, float]] = None,
) -> ObjectRoleMap:
    """Directly specify roles -- used in tests and SafeLIBERO templates."""
    return ObjectRoleMap(
        target=target,
        protected=protected,
        non_target=non_target,
        irrelevant=irrelevant or [],
        assignment_method="manual",
        weight_overrides=weight_overrides or {},
    )
