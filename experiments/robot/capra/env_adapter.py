"""Unified environment adapter for LIBERO / SafeLIBERO.

Wraps `libero.libero.envs.OffScreenRenderEnv` and exposes a minimal
step/reset interface that is consistent across LIBERO task suites and
the four CAPRA side-effect templates.

Phase 1 note: interface only – implementation in Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    task_suite_name: str = "libero_spatial"
    resolution: int = 256
    seed: int = 0
    use_safe_libero: bool = False        # swap in SafeLIBERO variant
    side_effect_template: Optional[str] = None  # one of CAPRA procedural templates


class CAPRAEnvAdapter:
    """Thin wrapper around a LIBERO env that adds CAPRA-needed hooks.

    Hooks added (stubs now, filled in Phase 2):
      - snapshot / restore  (see snapshot.py)
      - state signal access (see state_api.py)
      - contact / impulse readout
    """

    def __init__(self, env, cfg: EnvConfig):
        self._env = env
        self.cfg = cfg

    # ------------------------------------------------------------------ proxy
    def reset(self):
        return self._env.reset()

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        return self._env.step(action)

    def set_init_state(self, state):
        return self._env.set_init_state(state)

    def get_observation(self):
        return self._env.get_observation()

    def seed(self, s: int):
        self._env.seed(s)

    # -------------------------------------------------------- CAPRA extensions
    def get_sim_state(self) -> Any:
        """Return raw simulator state for snapshot/restore."""
        raise NotImplementedError("Implemented in Phase 2 via snapshot.py")

    def set_sim_state(self, state: Any) -> None:
        """Restore simulator to a previously captured state."""
        raise NotImplementedError("Implemented in Phase 2 via snapshot.py")

    def get_contact_info(self) -> Dict[str, Any]:
        """Return contact/impulse data from the physics engine."""
        raise NotImplementedError("Implemented in Phase 2 via state_api.py")

    def get_object_poses(self) -> Dict[str, np.ndarray]:
        """Return {object_name: pose_array} for all scene objects."""
        raise NotImplementedError("Implemented in Phase 2 via state_api.py")


def make_capra_env(task, cfg: EnvConfig) -> Tuple[CAPRAEnvAdapter, str]:
    """Factory: build a LIBERO env and wrap it with CAPRAEnvAdapter."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import os

    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": cfg.resolution,
        "camera_widths": cfg.resolution,
    }
    raw_env = OffScreenRenderEnv(**env_args)
    raw_env.seed(cfg.seed)
    return CAPRAEnvAdapter(raw_env, cfg), task_description
