# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Joint reset adapted from the pinned AME-2 teacher task."""
import torch

from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields


def reset_joints_by_scale(env, env_ids, position_range=(.5, 1.5)):
  ids = resolve_env_ids(env, env_ids)
  robot = env.scene["robot"]
  positions = robot.data.default_joint_pos[ids].clone()
  positions *= torch.empty_like(positions).uniform_(*position_range)
  limits = robot.data.soft_joint_pos_limits[ids]
  positions = positions.clamp(limits[..., 0], limits[..., 1])
  robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=ids)


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def randomize_mass_with_inertia(env, env_ids, ranges, asset_cfg):
  """Add the reference payload mass and scale inertia from the same defaults.

  AME-2's mass randomizer defaults to recompute_inertia=True. mjlab's
  dr.body_mass changes mass alone, so it cannot replace this source behavior.
  """
  ids = resolve_env_ids(env, env_ids)
  robot = env.scene[asset_cfg.name]
  bodies = robot.data.indexing.body_ids[asset_cfg.body_ids]
  default_mass = env.sim.get_default_field("body_mass")[bodies]
  default_inertia = env.sim.get_default_field("body_inertia")[bodies]
  delta = torch.empty(len(ids), len(bodies), device=env.device).uniform_(*ranges)
  mass = (default_mass + delta).clamp(min=1e-6)
  env.sim.model.body_mass[ids[:, None], bodies] = mass
  env.sim.model.body_inertia[ids[:, None], bodies] = (
    default_inertia[None] * (mass / default_mass)[..., None]
  )
