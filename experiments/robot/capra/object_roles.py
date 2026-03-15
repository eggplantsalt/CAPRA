"""Object role taxonomy for CAPRA footprint computation.

Every object in the scene is assigned one of four roles at the start
of each task. Only roles are used in footprint weighting; raw object
names are not hard-coded anywhere in the training loop.

Roles
-----
TARGET          : the object the robot is currently instructed to manipulate
PROTECTED       : objects that must not be displaced (side-effect penalty)
NON_TARGET      : other movable objects (smaller penalty)
IRRELEVANT     : fixed furniture / walls (ignored in footprint)

Phase 1: role dataclass + assignment stub.
Phase 2: auto-assign from task BDDL + SafeLIBERO metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class ObjectRole(Enum):
    TARGET = auto()
    PROTECTED = auto()
    NON_TARGET = auto()
    IRRELEVANT = auto()


@dataclass
class ObjectRoleMap:
    """Mapping from object name to role for one task episode."""
    target: List[str] = field(default_factory=list)
    protected: List[str] = field(default_factory=list)
    non_target: List[str] = field(default_factory=list)
    irrelevant: List[str] = field(default_factory=list)

    def get_role(self, name: str) -> ObjectRole:
        if name in self.target:
            return ObjectRole.TARGET
        if name in self.protected:
            return ObjectRole.PROTECTED
        if name in self.non_target:
            return ObjectRole.NON_TARGET
        return ObjectRole.IRRELEVANT

    def all_names(self) -> List[str]:
        return self.target + self.protected + self.non_target + self.irrelevant


def assign_roles_from_task(task_description: str, object_names: List[str]) -> ObjectRoleMap:
    """Assign object roles for a task.

    Phase 2: parse task BDDL / SafeLIBERO metadata to identify the
    manipulation target and mark remaining objects as protected or
    non-target based on proximity and task stage.

    For now raises NotImplementedError to make the dependency explicit.
    """
    raise NotImplementedError(
        "Phase 2: parse task BDDL + SafeLIBERO object metadata."
    )
