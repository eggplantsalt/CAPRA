"""Tests for footprint.py signals.py -- pure numpy, no env required.

Covers:
- Each of the three footprint components can be triggered independently.
- TARGET objects are never charged.
- PROTECTED has higher weight than NON_TARGET.
- Footprint scalar formula matches manual calculation.
- Progress and footprint are decoupled objects.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.robot.capra.core.capra_config import CAPRAConfig
from experiments.robot.capra.scene.object_roles import assign_roles_manual
from experiments.robot.capra.core.signals import (
    ObjectPose, ContactEvent, SupportRelation, StateSignals,
    read_topple_flags, check_workspace_violations,
)
from experiments.robot.capra.core.footprint import (
    FootprintComponents, aggregate_footprint_components, compute_footprint,
    compute_non_target_displacement, compute_contact_impulse,
    compute_irreversible_events,
)
from experiments.robot.capra.scene.task_progress import (
    ProgressResult, libero_info_progress, pick_height_proxy,
    compute_progress_from_rollout,
)


def pose(name, pos, quat=None):
    if quat is None:
        quat = np.array([0., 0., 0., 1.])
    return ObjectPose(name=name, position=np.array(pos, float),
                      orientation=np.array(quat, float))


def sigs(step, poses=None, contacts=None, support_relations=None,
         topple_flags=None, workspace_violations=None):
    return StateSignals(
        step=step,
        object_poses=poses or {},
        contacts=contacts or [],
        support_relations=support_relations or [],
        topple_flags=topple_flags or {},
        workspace_violations=workspace_violations or {},
    )


# ---- Component 1: displacement ------------------------------------------

class TestDisplacementComponent:
    def test_non_target_charged(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=["plate"])
        before = sigs(0, poses={"mug": pose("mug",[0,0,.9]),
                                "cup": pose("cup",[.1,0,.9]),
                                "plate": pose("plate",[.2,0,.9])})
        after  = sigs(1, poses={"mug": pose("mug",[.1,0,.9]),    # target -- free
                                "cup": pose("cup",[.1,0,.9]),    # protected, no move
                                "plate": pose("plate",[.25,0,.9])})  # 0.05 m
        total, per = compute_non_target_displacement(before, after, rm)
        assert total == pytest.approx(0.05, abs=1e-6)
        assert "plate" in per
        assert "mug" not in per

    def test_target_never_charged(self):
        rm = assign_roles_manual(target=["mug"], protected=[], non_target=[])
        before = sigs(0, poses={"mug": pose("mug",[0,0,.9])})
        after  = sigs(1, poses={"mug": pose("mug",[1,0,.9])})
        total, _ = compute_non_target_displacement(before, after, rm)
        assert total == pytest.approx(0.0)

    def test_protected_weight_gt_non_target(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=["plate"])
        before = sigs(0, poses={"cup":   pose("cup",  [0,0,.9]),
                                "plate": pose("plate",[.3,0,.9])})
        after  = sigs(1, poses={"cup":   pose("cup",  [.1,0,.9]),  # 0.1m
                                "plate": pose("plate",[.4,0,.9])})  # 0.1m
        _, per = compute_non_target_displacement(before, after, rm)
        assert per["cup"] > per["plate"]

    def test_no_move_zero_cost(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=["plate"])
        s = sigs(0, poses={"cup": pose("cup",[.1,0,.9]), "plate": pose("plate",[.2,0,.9])})
        total, _ = compute_non_target_displacement(s, s, rm)
        assert total == pytest.approx(0.0)


# ---- Component 2: contact impulse ---------------------------------------

class TestContactImpulseComponent:
    def test_impulse_on_protected_charged(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=["plate"])
        s = sigs(1, contacts=[ContactEvent("cup","robot",0.5)])
        total, per = compute_contact_impulse(s, rm)
        assert total > 0.0
        assert "cup" in per

    def test_impulse_on_non_target_not_charged(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=["plate"])
        s = sigs(1, contacts=[ContactEvent("plate","robot",0.5)])
        total, _ = compute_contact_impulse(s, rm)
        assert total == pytest.approx(0.0)

    def test_no_contacts_zero(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=[])
        s = sigs(1, contacts=[])
        total, _ = compute_contact_impulse(s, rm)
        assert total == pytest.approx(0.0)


# ---- Component 3: irreversible events -----------------------------------

class TestIrreversibleComponent:
    def test_topple_triggers(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=[])
        before = sigs(0, topple_flags={"cup": False})
        after  = sigs(1, topple_flags={"cup": True})
        w, n_t, n_s, n_w = compute_irreversible_events(before, after, rm)
        assert n_t == 1
        assert w > 0.0

    def test_support_break_triggers(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=[])
        before = sigs(0,
            poses={"cup": pose("cup",[.2,0,.92]), "shelf": pose("shelf",[.2,0,.90])},
            support_relations=[SupportRelation("cup","shelf")])
        after  = sigs(1,
            poses={"cup": pose("cup",[.2,0,.80]), "shelf": pose("shelf",[.2,0,.90])},
            support_relations=[])   # relation gone, cup fell 12 cm
        w, n_t, n_s, n_w = compute_irreversible_events(before, after, rm)
        assert n_s == 1
        assert w > 0.0

    def test_workspace_violation_triggers(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=[])
        before = sigs(0, workspace_violations={"cup": False})
        after  = sigs(1, workspace_violations={"cup": True})
        w, n_t, n_s, n_w = compute_irreversible_events(before, after, rm)
        assert n_w == 1
        assert w > 0.0

    def test_each_component_independent(self):
        """Displacement only doesn't trigger impulse or irreversible."""
        rm = assign_roles_manual(target=["mug"], protected=[], non_target=["plate"])
        before = sigs(0, poses={"plate": pose("plate",[.2,0,.9])},
                      topple_flags={"plate": False}, workspace_violations={"plate": False})
        after  = sigs(1, poses={"plate": pose("plate",[.3,0,.9])},
                      contacts=[],
                      topple_flags={"plate": False}, workspace_violations={"plate": False})
        comps = aggregate_footprint_components(before, after, rm)
        assert comps.non_target_displacement > 0.0
        assert comps.contact_impulse == pytest.approx(0.0)
        assert comps.topple_count == 0


# ---- Scalar footprint ---------------------------------------------------

class TestScalarFootprint:
    def test_formula(self):
        cfg = CAPRAConfig(alpha_d=1.0, alpha_i=1.0, alpha_r=2.0)
        c = FootprintComponents(non_target_displacement=0.1,
                                contact_impulse=0.2,
                                irreversible_count=1.0)
        assert compute_footprint(c, cfg) == pytest.approx(1.0*0.1 + 1.0*0.2 + 2.0*1.0)

    def test_zero_when_nothing_moves(self):
        rm = assign_roles_manual(target=["mug"], protected=["cup"], non_target=["plate"])
        s = sigs(0, poses={"mug": pose("mug",[0,0,.9]),
                           "cup": pose("cup",[.1,0,.9]),
                           "plate": pose("plate",[.2,0,.9])})
        c = aggregate_footprint_components(s, s, rm)
        assert compute_footprint(c, CAPRAConfig()) == pytest.approx(0.0)


# ---- Decoupling ---------------------------------------------------------

class TestDecoupling:
    def test_same_progress_different_footprint(self):
        """Two actions have equal progress but different side-effects."""
        rm  = assign_roles_manual(target=["mug"], protected=["cup"], non_target=[])
        cfg = CAPRAConfig(alpha_d=1.0, alpha_i=0.0, alpha_r=0.0)

        info_b = {"num_satisfied_predicates": 0, "num_predicates": 2}
        info_a = {"num_satisfied_predicates": 1, "num_predicates": 2}
        prog_safe   = compute_progress_from_rollout({}, info_b, {}, info_a, "pick mug")
        prog_unsafe = compute_progress_from_rollout({}, info_b, {}, info_a, "pick mug")
        assert prog_safe.value == prog_unsafe.value == pytest.approx(0.5)

        # safe: cup not moved; unsafe: cup displaced
        s_before = sigs(0, poses={"mug": pose("mug",[0,0,.9]),
                                  "cup": pose("cup",[.1,0,.9])})
        s_safe   = sigs(1, poses={"mug": pose("mug",[0,0,1.0]),
                                  "cup": pose("cup",[.1,0,.9])})
        s_unsafe = sigs(1, poses={"mug": pose("mug",[0,0,1.0]),
                                  "cup": pose("cup",[.3,0,.9])})
        f_safe   = compute_footprint(aggregate_footprint_components(s_before, s_safe,   rm), cfg)
        f_unsafe = compute_footprint(aggregate_footprint_components(s_before, s_unsafe, rm), cfg)
        assert f_safe == pytest.approx(0.0)
        assert f_unsafe > f_safe


# ---- Task progress standalone -------------------------------------------

class TestTaskProgress:
    def test_info_progress_exact(self):
        info = {"num_satisfied_predicates": 1, "num_predicates": 4}
        assert libero_info_progress({}, "pick mug", info) == pytest.approx(0.25)

    def test_info_progress_missing_returns_minus_one(self):
        assert libero_info_progress({}, "pick mug", {}) == -1.0

    def test_pick_height_at_table_zero(self):
        obs = {"mug_pos": np.array([0., 0., 0.82])}
        assert pick_height_proxy(obs, "mug", lift_height_m=0.10, table_z=0.82) == pytest.approx(0.0)

    def test_pick_height_fully_lifted(self):
        obs = {"mug_pos": np.array([0., 0., 0.93])}
        assert pick_height_proxy(obs, "mug", lift_height_m=0.10, table_z=0.82) == pytest.approx(1.0)

    def test_progress_from_rollout_exact(self):
        ib = {"num_satisfied_predicates": 0, "num_predicates": 4}
        ia = {"num_satisfied_predicates": 2, "num_predicates": 4}
        r = compute_progress_from_rollout({}, ib, {}, ia, "pick mug")
        assert r.value == pytest.approx(0.5)
        assert r.is_approximate is False
        assert r.method == "libero_info"

    def test_progress_result_has_no_footprint_fields(self):
        r = ProgressResult(value=0.5, stage_before=0, stage_after=1, max_stages=2)
        assert not hasattr(r, "non_target_displacement")
        assert not hasattr(r, "contact_impulse")
