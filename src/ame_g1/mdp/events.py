"""AME-specific event functions."""

from __future__ import annotations

import torch


def resample_map_scan_drift(
  env,
  env_ids: torch.Tensor | None,
  std_xy: float = 0.02,
  attr_name: str = "_ame_map_scan_drift_xy",
) -> None:
  """Sample a per-environment XY drift for the robot-centric terrain map."""

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  if not hasattr(env, attr_name):
    setattr(env, attr_name, torch.zeros(env.num_envs, 2, device=env.device))
  drift = getattr(env, attr_name)
  drift[env_ids] = torch.randn(len(env_ids), 2, device=env.device) * std_xy
