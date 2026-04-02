"""Register AME Unitree G1 tasks."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from ame_g1.config.g1.env_cfgs import unitree_g1_ame_env_cfg
from ame_g1.config.g1.rl_cfg import unitree_g1_ame_ppo_runner_cfg
from ame_g1.config.variants import iter_task_specs
from ame_g1.learning.runner import AmeOnPolicyRunner

_REGISTERED = False


def register_tasks() -> None:
  """Register all AME Unitree G1 variant/phase tasks exactly once."""

  global _REGISTERED
  if _REGISTERED:
    return

  for task_spec in iter_task_specs():
    register_mjlab_task(
      task_id=task_spec.task_id,
      env_cfg=unitree_g1_ame_env_cfg(
        phase=task_spec.phase,
        variant=task_spec.variant,
        play=False,
      ),
      play_env_cfg=unitree_g1_ame_env_cfg(
        phase=task_spec.phase,
        variant=task_spec.variant,
        play=True,
      ),
      rl_cfg=unitree_g1_ame_ppo_runner_cfg(
        phase=task_spec.phase,
        variant=task_spec.variant,
      ),
      runner_cls=AmeOnPolicyRunner,
    )

  _REGISTERED = True
