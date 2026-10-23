# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit teacher curricula, adapted from leggedrobotics/ame2_minimal."""

import math

import torch


def ame2_terrain_levels_goalreaching(env, env_ids, command_name, move_up_dist=.5, move_down_dist=4.):
  if not hasattr(env, "success_rate_reach_ema"):
    env.success_rate_reach_ema = torch.zeros(env.num_envs, device=env.device)
  distance = env.command_manager.get_command(command_name)[env_ids, :2].norm(dim=1)
  remaining = (env.max_episode_length - env.episode_length_buf[env_ids]) * env.step_dt
  success = (distance < move_up_dist) & (remaining < 4.)
  env.success_rate_reach_ema[env_ids] = .8 * env.success_rate_reach_ema[env_ids] + .2 * success.float()
  promote = (distance < move_up_dist) & (env.success_rate_reach_ema[env_ids] > .5)
  demote = distance > move_down_dist
  env.scene.terrain.update_env_origins(env_ids, promote, demote)
  return {
    "terrain_level": env.scene.terrain.terrain_levels.float().mean(),
    "success_rate_ema": env.success_rate_reach_ema.mean(),
  }


def ame2_teacher_schedule(env, env_ids, max_steps=48000, command_name="base_pose", reset_event_name="reset_base"):
  """Expand starting yaw, goal cone and map-height noise over the first 2000 updates."""
  del env_ids
  progress = min(env.common_step_counter / max_steps, 1.)
  command = env.command_manager.get_term(command_name)
  command.cfg.cone_angle_deg = 10. + 35. * progress
  reset_cfg = env.event_manager.get_term_cfg(reset_event_name)
  reset_cfg.params["pose_range"]["yaw"] = (-math.pi * progress, math.pi * progress)
  mapping_cfg = env.observation_manager.get_term_cfg("teacher_mapping", "teacher_mapping")
  old_noise = mapping_cfg.params["custom_noise"]
  mapping_cfg.params["custom_noise"] = (old_noise[0], old_noise[1], .05 * progress)
  return {"yaw_limit": math.pi * progress, "goal_cone_angle": command.cfg.cone_angle_deg, "height_noise": .05 * progress}
