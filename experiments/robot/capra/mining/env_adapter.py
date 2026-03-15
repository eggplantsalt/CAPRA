# ===== CAPRA 环境适配器 (env_adapter.py) =====
#
# 作用
# ----
# 对 LIBERO 的 OffScreenRenderEnv 做薄包装，
# 统一暴露 CAPRA 需要的接口，不修改 LIBERO 代码。
#
# 主要功能
# --------
#   step/reset/seed/get_observation  代理到底层 LIBERO env（原样透传）
#   apply_procedural_template()      reset 后调用，注入程序化场景扰动
#   get_sim_state() / set_sim_state() MuJoCo 快照/还原（CF rollout 用）
#   get_object_poses()               返回所有可动物体的当前位置
#
# 访问 MuJoCo 的路径
# ------------------
#   env._env.sim  →  OffScreenRenderEnv 内部的 MjSim
#   sim.data.qpos, sim.data.body_xpos  →  物体状态
#   sim.model.body_name2id()  →  按名称找物体编号
#
# make_capra_env() 工厂函数
# --------------------------
#   从 LIBERO task 对象 + EnvConfig 构建一个 CAPRAEnvAdapter
#   使用方式：
#     adapter, task_desc = make_capra_env(task, EnvConfig(task_suite_name='libero_spatial'))

"""Unified environment adapter for LIBERO / SafeLIBERO.

Wraps `libero.libero.envs.OffScreenRenderEnv` and exposes a minimal
step/reset interface consistent across LIBERO task suites and the four
CAPRA side-effect templates.

Phase 8: apply_procedural_template() hook implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    task_suite_name: str        = "libero_spatial"
    resolution: int             = 256
    seed: int                   = 0
    use_safe_libero: bool       = False
    side_effect_template: Optional[str] = None  # SideEffectTemplate value


class CAPRAEnvAdapter:
    """Thin wrapper around a LIBERO env that adds CAPRA-needed hooks.

    Added in Phase 8:
      - apply_procedural_template(): reset-time scene perturbation

    Stubs for Phase 9:
      - snapshot / restore  (see snapshot.py)
      - contact / impulse readout (see state_api.py)
    """

    def __init__(self, env, cfg: EnvConfig):
        self._env = env
        self.cfg = cfg
        self._last_template_meta = None   # TemplateMetadata from last reset

    # ----------------------------------------------------------------- proxy
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

    # ------------------------------------------------------ CAPRA Phase 8 hook
    def apply_procedural_template(
        self,
        obs: Dict[str, Any],
        task_description: str = "",
        task_id: int = 0,
        episode_idx: int = 0,
        template_cfg=None,
    ):
        """Apply a procedural side-effect template to the current scene.

        Must be called AFTER reset() / set_init_state() and BEFORE first step.

        Parameters
        ----------
        obs              : observation dict from reset()
        task_description : language instruction string
        task_id          : base task index in suite
        episode_idx      : rollout index
        template_cfg     : TemplateConfig; if None, built from self.cfg.side_effect_template

        Returns
        -------
        TemplateMetadata  -- perturbation details for logging
        None              -- if side_effect_template is not set and template_cfg is None
        """
        from experiments.robot.capra.eval.procedural_splits import (
            SideEffectTemplate,
            TemplateConfig,
            apply_template_to_env,
        )
        import numpy as np

        if template_cfg is None:
            tmpl_str = self.cfg.side_effect_template
            if tmpl_str is None:
                return None
            template_cfg = TemplateConfig(
                template=SideEffectTemplate(tmpl_str),
                base_task_suite=self.cfg.task_suite_name,
                seed=self.cfg.seed,
            )

        rng = np.random.default_rng(template_cfg.seed + episode_idx * 1000 + task_id)
        meta = apply_template_to_env(
            env=self._env,   # pass raw env so _get_sim can reach .sim
            obs=obs,
            cfg=template_cfg,
            task_description=task_description,
            task_id=task_id,
            episode_idx=episode_idx,
            rng=rng,
        )
        self._last_template_meta = meta
        return meta

    @property
    def last_template_meta(self):
        """TemplateMetadata from the most recent apply_procedural_template call."""
        return self._last_template_meta

    # -------------------------------------------------- CAPRA future stubs
    def get_sim_state(self) -> Any:
        """Return raw simulator state for snapshot/restore (Phase 9)."""
        raw = getattr(self._env, "sim", None)
        if raw is not None and hasattr(raw, "get_state"):
            return raw.get_state()
        raise NotImplementedError("sim.get_state() not available on this env")

    def set_sim_state(self, state: Any) -> None:
        """Restore simulator state (Phase 9)."""
        raw = getattr(self._env, "sim", None)
        if raw is not None and hasattr(raw, "set_state"):
            raw.set_state(state)
            raw.forward()
            return
        raise NotImplementedError("sim.set_state() not available on this env")

    def get_contact_info(self) -> Dict[str, Any]:
        """Return contact/impulse data (Phase 9 via state_api.py)."""
        raise NotImplementedError

    def get_object_poses(self) -> Dict[str, np.ndarray]:
        """Return {object_name: xpos} for all movable bodies."""
        from experiments.robot.capra.eval.procedural_splits import (
            _get_sim, _list_movable_bodies, _get_body_xpos
        )
        sim = _get_sim(self._env)
        if sim is None:
            return {}
        return {
            name: _get_body_xpos(sim, name)
            for name in _list_movable_bodies(sim)
        }


def make_capra_env(task, cfg: EnvConfig) -> Tuple[CAPRAEnvAdapter, str]:
    """Factory: build a LIBERO env wrapped with CAPRAEnvAdapter."""
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
        "camera_widths":  cfg.resolution,
    }
    raw_env = OffScreenRenderEnv(**env_args)
    raw_env.seed(cfg.seed)
    return CAPRAEnvAdapter(raw_env, cfg), task_description
