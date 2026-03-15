"""Phase 8 smoke tests: procedural side-effect templates.

All tests are pure-Python (no GPU, no LIBERO import required).
We use a mock env/sim to exercise the full apply_template_to_env path.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from experiments.robot.capra.eval.procedural_splits import (
    DEFAULT_CONFIGS,
    SideEffectTemplate,
    TEMPLATE_SUMMARY,
    TemplateConfig,
    TemplateMetadata,
    apply_template_to_env,
    build_role_map_for_template,
    get_template_config,
    list_all_templates,
    save_template_metadata,
    _get_sim,
    _set_body_xpos,
    _get_body_xpos,
    _list_movable_bodies,
    _find_target_body,
)


# ===========================================================================
# Mock sim / env
# ===========================================================================

class MockModel:
    def __init__(self, bodies):
        self._names      = [b[0] for b in bodies]
        self.nbody       = len(bodies)
        self.body_jntadr = [i if b[1] else -1 for i, b in enumerate(bodies)]
        self.jnt_type    = [0] * len(bodies)

    def body_name2id(self, name): return self._names.index(name)
    def body_id2name(self, i):    return self._names[i]


class MockSim:
    def __init__(self, bodies):
        self.model = MockModel(bodies)
        self.data  = type('D', (), {
            'qpos':      np.zeros(len(bodies) * 7),
            'body_xpos': np.zeros((len(bodies), 3)),
        })()
        self._fwd = 0

    def forward(self): self._fwd += 1


class MockEnv:
    def __init__(self, bodies): self.sim = MockSim(bodies)


BODIES = [
    ("mug_1",   True),
    ("bowl_1",  True),
    ("plate_1", True),
    ("box_1",   True),
    ("robot0_base", False),
]
TASK_DESC = "pick the mug_1 and place it on the plate"


def _obs(body_names):
    obs = {}
    for i, name in enumerate(body_names):
        obs[f"{name}_pos"]  = np.array([0.1 * i, 0.0, 0.05])
        obs[f"{name}_quat"] = np.array([0.0, 0.0, 0.0, 1.0])
    return obs


def _run(template, **kw):
    env  = MockEnv(BODIES)
    obs  = _obs([b[0] for b in BODIES])
    cfg  = get_template_config(template, seed=42, **kw)
    meta = apply_template_to_env(env=env, obs=obs, cfg=cfg,
                                  task_description=TASK_DESC,
                                  task_id=0, episode_idx=0)
    return meta, env


# ===========================================================================
# Helper tests
# ===========================================================================

def test_get_sim_direct():
    env = MockEnv(BODIES)
    assert _get_sim(env) is env.sim

def test_get_sim_nested():
    class Outer:
        _env = MockEnv(BODIES)
    assert _get_sim(Outer()) is Outer._env.sim

def test_get_sim_none():
    assert _get_sim(object()) is None

def test_list_movable_bodies():
    env   = MockEnv(BODIES)
    names = _list_movable_bodies(env.sim)
    assert "mug_1" in names
    assert "robot0_base" not in names

def test_find_target_body():
    env  = MockEnv(BODIES)
    obs  = _obs([b[0] for b in BODIES])
    tgt  = _find_target_body(env.sim, obs, TASK_DESC)
    assert tgt == "mug_1"

def test_set_body_xpos_exact():
    env = MockEnv(BODIES)
    f   = _set_body_xpos(env.sim, "mug_1", np.array([0.5, 0.1, 0.05]))
    assert f == "exact"
    np.testing.assert_allclose(env.sim.data.qpos[0:3], [0.5, 0.1, 0.05])

def test_set_body_xpos_none_sim():
    assert _set_body_xpos(None, "mug_1", np.array([0.0, 0.0, 0.0])) == "none"


# ===========================================================================
# Config / metadata
# ===========================================================================

def test_default_configs_all_templates():
    for t in SideEffectTemplate:
        assert DEFAULT_CONFIGS[t.value].template == t

def test_get_template_config_override():
    cfg = get_template_config(SideEffectTemplate.CHAIN_REACTION, chain_length=5)
    assert cfg.chain_length == 5

def test_list_all_templates():
    templates = list_all_templates()
    assert len(templates) == 4
    assert SideEffectTemplate.COLLATERAL_CLUTTER in templates

def test_template_summary_all_keys():
    required = {"initial_conditions", "side_effect_exposed",
                "footprint_signals", "fidelity_note", "implementation_limit"}
    for t in SideEffectTemplate:
        assert required == set(TEMPLATE_SUMMARY[t.value].keys()), t

def test_template_metadata_roundtrip(tmp_path):
    meta = TemplateMetadata(
        template=SideEffectTemplate.COLLATERAL_CLUTTER.value,
        base_task_suite="libero_spatial",
        base_task_id=2, episode_idx=7,
        config=asdict(TemplateConfig()),
        perturbation_fidelity="exact",
        perturbed_object_names=["bowl_1", "plate_1"],
        perturbed_positions={"bowl_1": [0.1, 0.2, 0.05]},
        hazard_object_names=["bowl_1"],
        footprint_signals_exposed=["protected_object_displacement"],
    )
    path = save_template_metadata(meta, tmp_path)
    loaded = TemplateMetadata.from_dict(json.loads(path.read_text()))
    assert loaded.template == meta.template
    assert loaded.perturbation_fidelity == "exact"
    assert loaded.perturbed_object_names == ["bowl_1", "plate_1"]


# ===========================================================================
# Template 1: Collateral Clutter
# ===========================================================================

def test_collateral_clutter_n_objects():
    meta, env = _run(SideEffectTemplate.COLLATERAL_CLUTTER, clutter_n_objects=2)
    assert meta.template == SideEffectTemplate.COLLATERAL_CLUTTER.value
    assert len(meta.perturbed_object_names) == 2
    assert meta.perturbation_fidelity == "exact"
    assert "protected_object_displacement" in meta.footprint_signals_exposed


def test_collateral_clutter_objects_were_moved():
    meta, env = _run(SideEffectTemplate.COLLATERAL_CLUTTER, clutter_n_objects=2)
    for name in meta.perturbed_object_names:
        bid  = env.sim.model.body_name2id(name)
        jadr = env.sim.model.body_jntadr[bid]
        pos  = env.sim.data.qpos[jadr:jadr + 3]
        assert not np.allclose(pos, [0.0, 0.0, 0.0]), f"{name} was not moved"


def test_collateral_clutter_proximity():
    meta, env = _run(SideEffectTemplate.COLLATERAL_CLUTTER,
                     clutter_n_objects=2, clutter_proximity_m=0.12)
    # mug_1 (target) is body 0, obs pos = [0, 0, 0.05]
    target_pos = np.array([0.0, 0.0, 0.05])
    for name, pos_list in meta.perturbed_positions.items():
        pos   = np.array(pos_list)
        dist  = float(np.linalg.norm(pos[:2] - target_pos[:2]))
        assert abs(dist - 0.12) < 0.02, f"{name} dist={dist:.4f} expected ~0.12"


def test_collateral_clutter_metadata_has_signals():
    meta, _ = _run(SideEffectTemplate.COLLATERAL_CLUTTER)
    assert "contact_impulse" in meta.footprint_signals_exposed


# ===========================================================================
# Template 2: Support-Critical Neighbor
# ===========================================================================

def test_support_critical_neighbor_stack_count():
    meta, _ = _run(SideEffectTemplate.SUPPORT_CRITICAL_NEIGHBOR, stack_height=2)
    assert len(meta.perturbed_object_names) == 2
    assert "topple_count" in meta.footprint_signals_exposed
    assert "support_break_count" in meta.footprint_signals_exposed


def test_support_critical_neighbor_stacked_heights():
    meta, env = _run(SideEffectTemplate.SUPPORT_CRITICAL_NEIGHBOR,
                     stack_height=2, stack_proximity_m=0.10)
    positions = [meta.perturbed_positions[n] for n in meta.perturbed_object_names]
    # Each block should be higher than the previous by ~0.045 m
    z_vals = [p[2] for p in positions]
    for i in range(1, len(z_vals)):
        assert z_vals[i] > z_vals[i - 1] - 0.001, "Stack not ascending in Z"


def test_support_critical_neighbor_x_offset():
    meta, _ = _run(SideEffectTemplate.SUPPORT_CRITICAL_NEIGHBOR, stack_proximity_m=0.10)
    # All stack objects should be at target.x + 0.10 (mug_1 x=0)
    for name, pos in meta.perturbed_positions.items():
        assert abs(pos[0] - 0.10) < 0.01, f"{name} x={pos[0]:.4f} expected ~0.10"


# ===========================================================================
# Template 3: Chain Reaction
# ===========================================================================

def test_chain_reaction_length():
    meta, _ = _run(SideEffectTemplate.CHAIN_REACTION, chain_length=3)
    assert len(meta.perturbed_object_names) == 3
    assert "topple_count" in meta.footprint_signals_exposed


def test_chain_reaction_spacing():
    meta, _ = _run(SideEffectTemplate.CHAIN_REACTION,
                   chain_length=3, chain_spacing_m=0.07,
                   chain_direction_x=1.0, chain_direction_y=0.0)
    positions = [meta.perturbed_positions[n] for n in meta.perturbed_object_names]
    # Objects should be spaced 0.07 m apart along x-axis
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        assert abs(dx - 0.07) < 0.01, f"spacing[{i}]={dx:.4f} expected ~0.07"


def test_chain_reaction_direction_normalised():
    # Non-unit direction should still produce correct spacing
    meta, _ = _run(SideEffectTemplate.CHAIN_REACTION,
                   chain_length=2, chain_spacing_m=0.07,
                   chain_direction_x=2.0, chain_direction_y=0.0)
    positions = [meta.perturbed_positions[n] for n in meta.perturbed_object_names]
    dx = abs(positions[1][0] - positions[0][0])
    assert abs(dx - 0.07) < 0.01, f"dx={dx:.4f}"


# ===========================================================================
# Template 4: Occluded Remembered Hazard
# ===========================================================================

def test_occluded_hazard_one_object():
    meta, _ = _run(SideEffectTemplate.OCCLUDED_REMEMBERED_HAZARD,
                   use_static_occluder=False)
    assert len(meta.perturbed_object_names) == 1
    assert "protected_object_displacement" in meta.footprint_signals_exposed


def test_occluded_hazard_with_occluder():
    meta, _ = _run(SideEffectTemplate.OCCLUDED_REMEMBERED_HAZARD,
                   use_static_occluder=True)
    assert len(meta.perturbed_object_names) == 2


def test_occluded_hazard_offset():
    meta, _ = _run(SideEffectTemplate.OCCLUDED_REMEMBERED_HAZARD,
                   occluder_offset_m=0.30, use_static_occluder=False)
    name = meta.perturbed_object_names[0]
    pos  = meta.perturbed_positions[name]
    # Hazard should be ~0.30 m away in Y from target (mug_1 at y=0)
    assert abs(abs(pos[1]) - 0.30) < 0.02, f"y offset={pos[1]:.4f}"


def test_occluded_hazard_config_fields():
    cfg = get_template_config(SideEffectTemplate.OCCLUDED_REMEMBERED_HAZARD,
                               reveal_step=20, occluder_offset_m=0.25)
    assert cfg.reveal_step == 20
    assert cfg.occluder_offset_m == 0.25


# ===========================================================================
# No-sim fallback (fidelity=none)
# ===========================================================================

def test_no_sim_returns_empty_perturbation():
    class NoSimEnv: pass
    env = NoSimEnv()
    obs = _obs([b[0] for b in BODIES])
    for t in SideEffectTemplate:
        cfg  = get_template_config(t, seed=0)
        meta = apply_template_to_env(env=env, obs=obs, cfg=cfg,
                                      task_description=TASK_DESC)
        assert meta.perturbation_fidelity == "none"
        assert meta.perturbed_object_names == []
        assert "sim not accessible" in meta.notes


# ===========================================================================
# Metadata export
# ===========================================================================

def test_save_all_templates_metadata(tmp_path):
    for t in SideEffectTemplate:
        meta, _ = _run(t)
        path = save_template_metadata(meta, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["template"] == t.value
        assert "perturbation_fidelity" in data
        assert "footprint_signals_exposed" in data
        assert "config" in data
        assert "perturbed_positions" in data


def test_metadata_config_field_is_dict():
    meta, _ = _run(SideEffectTemplate.CHAIN_REACTION, chain_length=3)
    assert isinstance(meta.config, dict)
    assert meta.config["chain_length"] == 3


# ===========================================================================
# build_role_map_for_template
# ===========================================================================

def test_build_role_map_protected_template():
    meta, _ = _run(SideEffectTemplate.SUPPORT_CRITICAL_NEIGHBOR, stack_height=2)
    role_map = build_role_map_for_template(
        meta=meta,
        base_object_names=[b[0] for b in BODIES],
        task_description=TASK_DESC,
    )
    for name in meta.hazard_object_names:
        assert name in role_map.protected


def test_build_role_map_non_target_template():
    meta, _ = _run(SideEffectTemplate.COLLATERAL_CLUTTER, clutter_n_objects=1)
    role_map = build_role_map_for_template(
        meta=meta,
        base_object_names=[b[0] for b in BODIES],
        task_description=TASK_DESC,
    )
    for name in meta.hazard_object_names:
        assert name in role_map.non_target


# ===========================================================================
# env_adapter integration
# ===========================================================================

def test_env_adapter_apply_template_no_sim():
    from experiments.robot.capra.mining.env_adapter import CAPRAEnvAdapter, EnvConfig

    class FakeRaw:
        pass  # no .sim

    cfg = EnvConfig(side_effect_template="collateral_clutter", seed=0)
    adapter = CAPRAEnvAdapter(FakeRaw(), cfg)
    obs = _obs([b[0] for b in BODIES])
    meta = adapter.apply_procedural_template(obs, task_description=TASK_DESC)
    assert meta is not None
    assert meta.template == "collateral_clutter"
    assert meta.perturbation_fidelity == "none"  # no sim


def test_env_adapter_no_template_returns_none():
    from experiments.robot.capra.mining.env_adapter import CAPRAEnvAdapter, EnvConfig

    class FakeRaw: pass
    cfg     = EnvConfig(side_effect_template=None)
    adapter = CAPRAEnvAdapter(FakeRaw(), cfg)
    obs     = _obs([b[0] for b in BODIES])
    assert adapter.apply_procedural_template(obs) is None

