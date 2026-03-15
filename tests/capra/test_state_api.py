"""Tests for state_api.py / signals.py -- pure numpy, no LIBERO install needed.

Covers:
- read_object_poses correctly parses obs dict keys.
- read_topple_flags detects toppling via quaternion change.
- check_workspace_violations flags out-of-bounds objects.
- read_support_relations detects vertical stacking.
- StateSignals workspace_violation property aggregates per-object flags.
- Signals are independent of task progress.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from experiments.robot.capra.core.signals import (
    ObjectPose,
    StateSignals,
    read_object_poses,
    read_topple_flags,
    read_support_relations,
    check_workspace_violations,
    read_state_signals,
    DEFAULT_WORKSPACE_BOUNDS,
)
from experiments.robot.capra.core.state_api import StateSignals as StateSignalsAlias


# ---------------------------------------------------------------------------
# Helper: build minimal obs dict
# ---------------------------------------------------------------------------

def make_obs(**objects):
    """Build a fake obs dict.

    Usage: make_obs(mug=([x,y,z], [qx,qy,qz,qw]), cup=([x,y,z], None))
    None quat defaults to identity.
    """
    obs = {}
    for name, (pos, quat) in objects.items():
        obs[f"{name}_pos"] = np.array(pos, dtype=float)
        if quat is None:
            quat = [0., 0., 0., 1.]
        obs[f"{name}_quat"] = np.array(quat, dtype=float)
    return obs


# ---------------------------------------------------------------------------
# re-export test
# ---------------------------------------------------------------------------

def test_state_api_reexports_signals():
    """state_api.StateSignals must be the same class as signals.StateSignals."""
    assert StateSignalsAlias is StateSignals


# ---------------------------------------------------------------------------
# read_object_poses
# ---------------------------------------------------------------------------

class TestReadObjectPoses:
    def test_basic_parse(self):
        obs = make_obs(
            mug=([0.1, 0.2, 0.9], None),
            cup=([0.3, 0.0, 0.9], [0., 0., 0., 1.]),
        )
        poses = read_object_poses(obs)
        assert "mug" in poses
        assert "cup" in poses
        np.testing.assert_allclose(poses["mug"].position, [0.1, 0.2, 0.9])
        np.testing.assert_allclose(poses["cup"].orientation, [0., 0., 0., 1.])

    def test_only_requested_names(self):
        obs = make_obs(
            mug=([0.1, 0.2, 0.9], None),
            cup=([0.3, 0.0, 0.9], None),
        )
        poses = read_object_poses(obs, object_names=["mug"])
        assert "mug" in poses
        assert "cup" not in poses

    def test_missing_quat_key_skipped(self):
        obs = {"mug_pos": np.array([0.1, 0.2, 0.9])}
        # No mug_quat key -- should be silently skipped
        poses = read_object_poses(obs)
        assert "mug" not in poses

    def test_non_pose_keys_ignored(self):
        obs = make_obs(mug=([0.1, 0.2, 0.9], None))
        obs["robot0_eef_pos"] = np.array([0., 0., 0.])
        obs["robot0_eef_quat"] = np.array([0., 0., 0., 1.])
        obs["some_other_key"] = 42
        poses = read_object_poses(obs)
        # robot0_eef would be parsed (ends in _pos + has _quat) -- that's fine
        # but non-pose numeric scalars should not crash
        assert "mug" in poses

    def test_is_approximate_false(self):
        obs = make_obs(mug=([0.0, 0.0, 0.9], None))
        poses = read_object_poses(obs)
        assert poses["mug"].is_approximate is False


# ---------------------------------------------------------------------------
# ObjectPose.tilt_angle_deg
# ---------------------------------------------------------------------------

class TestTiltAngle:
    def test_identity_zero_tilt(self):
        p = ObjectPose("obj", np.zeros(3), np.array([0., 0., 0., 1.]))
        assert p.tilt_angle_deg() == pytest.approx(0.0, abs=1e-5)

    def test_90_deg_rotation(self):
        # 90-degree rotation around x: quat = [sin(45), 0, 0, cos(45)]
        s = math.sin(math.radians(45))
        c = math.cos(math.radians(45))
        p = ObjectPose("obj", np.zeros(3), np.array([s, 0., 0., c]))
        assert p.tilt_angle_deg() == pytest.approx(90.0, abs=1.0)


# ---------------------------------------------------------------------------
# read_topple_flags
# ---------------------------------------------------------------------------

class TestReadToppleFlags:
    def _pose(self, name, quat):
        return ObjectPose(name, np.zeros(3), np.array(quat, float))

    def test_upright_not_toppled(self):
        q = [0., 0., 0., 1.]
        before = {"mug": self._pose("mug", q)}
        after  = {"mug": self._pose("mug", q)}
        flags = read_topple_flags(before, after)
        assert flags["mug"] is False

    def test_large_rotation_toppled(self):
        q_up = [0., 0., 0., 1.]
        # 90-deg rotation around x
        s = math.sin(math.radians(45))
        c = math.cos(math.radians(45))
        q_tipped = [s, 0., 0., c]
        before = {"mug": self._pose("mug", q_up)}
        after  = {"mug": self._pose("mug", q_tipped)}
        flags = read_topple_flags(before, after,
                                  angle_change_threshold_deg=45.0,
                                  absolute_tilt_threshold_deg=60.0)
        assert flags["mug"] is True

    def test_missing_before_not_toppled(self):
        q = [0., 0., 0., 1.]
        before = {}
        after  = {"mug": self._pose("mug", q)}
        flags = read_topple_flags(before, after)
        assert flags["mug"] is False


# ---------------------------------------------------------------------------
# read_support_relations
# ---------------------------------------------------------------------------

class TestReadSupportRelations:
    def _poses(self, **items):
        return {
            name: ObjectPose(name, np.array(pos, float), np.array([0.,0.,0.,1.]))
            for name, pos in items.items()
        }

    def test_detects_stacking(self):
        poses = self._poses(
            plate=[0.2, 0.0, 0.90],
            cup  =[0.2, 0.0, 0.94],   # cup is 4cm above plate, 0 xy offset
        )
        rels = read_support_relations(poses,
                                      vertical_gap_m=0.06,
                                      horizontal_gap_m=0.08)
        assert any(r.supported == "cup" and r.supporter == "plate" for r in rels)

    def test_no_relation_when_far_apart(self):
        poses = self._poses(
            plate=[0.2, 0.0, 0.90],
            cup  =[0.8, 0.0, 0.91],   # far away horizontally
        )
        rels = read_support_relations(poses)
        assert len(rels) == 0

    def test_all_approximate(self):
        poses = self._poses(
            plate=[0.2, 0.0, 0.90],
            cup  =[0.2, 0.0, 0.94],
        )
        rels = read_support_relations(poses)
        assert all(r.is_approximate for r in rels)


# ---------------------------------------------------------------------------
# check_workspace_violations
# ---------------------------------------------------------------------------

class TestWorkspaceViolations:
    def _pose(self, name, pos):
        return ObjectPose(name, np.array(pos, float), np.array([0.,0.,0.,1.]))

    def test_in_bounds_not_violated(self):
        poses = {"mug": self._pose("mug", [0.0, 0.0, 0.9])}
        viols = check_workspace_violations(poses)
        assert viols["mug"] is False

    def test_out_of_x_bounds_violated(self):
        poses = {"mug": self._pose("mug", [1.0, 0.0, 0.9])}
        viols = check_workspace_violations(poses)
        assert viols["mug"] is True

    def test_below_z_floor_violated(self):
        poses = {"mug": self._pose("mug", [0.0, 0.0, 0.5])}
        viols = check_workspace_violations(poses)
        assert viols["mug"] is True

    def test_custom_bounds(self):
        bounds = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 2.0)}
        poses = {"mug": self._pose("mug", [0.05, 0.0, 1.0])}
        viols = check_workspace_violations(poses, bounds=bounds)
        assert viols["mug"] is False


# ---------------------------------------------------------------------------
# StateSignals.workspace_violation property
# ---------------------------------------------------------------------------

def test_state_signals_workspace_violation_property():
    s_ok   = StateSignals(step=0, workspace_violations={"mug": False, "cup": False})
    s_viol = StateSignals(step=1, workspace_violations={"mug": False, "cup": True})
    assert s_ok.workspace_violation is False
    assert s_viol.workspace_violation is True


# ---------------------------------------------------------------------------
# read_state_signals (no env)
# ---------------------------------------------------------------------------

def test_read_state_signals_no_env():
    obs = make_obs(
        mug=([0.0, 0.0, 0.9], None),
        cup=([0.1, 0.0, 0.9], None),
    )
    signals = read_state_signals(obs, step=3, env=None)
    assert signals.step == 3
    assert "mug" in signals.object_poses
    assert "cup" in signals.object_poses
    assert signals.contacts == []   # no env, no contacts
    assert signals.raw_obs is obs
