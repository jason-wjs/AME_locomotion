# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""AME-2 teacher observations, adapted from leggedrobotics/ame2_minimal."""

from __future__ import annotations

import torch


def ame2_base_lin_vel(env):
  """Root COM velocity expressed in the root link frame, matching Isaac Lab."""
  return env.scene["robot"].data.root_com_lin_vel_b


def ame2_base_ang_vel(env):
  return env.scene["robot"].data.root_link_ang_vel_b


def ame2_projected_gravity(env):
  return env.scene["robot"].data.projected_gravity_b


def _height_samples(env, sensor_name, offset, void_bias_range):
  sensor = env.scene[sensor_name]
  z = sensor.data.hit_pos_w[..., 2] - sensor.data.pos_w[:, 2, None] + offset
  valid = (sensor.data.distances >= 0) & torch.isfinite(z)
  minimum = torch.where(valid, z, float("inf")).amin(dim=1, keepdim=True)
  minimum = torch.where(valid.any(dim=1, keepdim=True), minimum, -10.0)
  bias = torch.empty((env.num_envs, 1), device=z.device).uniform_(*void_bias_range)
  return torch.where(valid, z, minimum + bias)


def _grid(env, sensor_name):
  cache_name = f"_ame2_grid_{sensor_name}"
  if not hasattr(env, cache_name):
    pattern = env.scene[sensor_name].cfg.pattern
    nx = int(round(pattern.size[0] / pattern.resolution)) + 1
    ny = int(round(pattern.size[1] / pattern.resolution)) + 1
    offsets, _ = pattern.generate_rays(None, env.device)
    # mjlab GridPatternCfg enumerates x fastest; AME-2 maps use [x,y,channels].
    xy = offsets[:, :2].reshape(ny, nx, 2).transpose(0, 1).contiguous()
    setattr(env, cache_name, (nx, ny, xy))
  return getattr(env, cache_name)


def ame2_height_scan(env, sensor_name, offset=0.74, void_bias_range=(-10.0, -10.0)):
  z = _height_samples(env, sensor_name, offset, void_bias_range)
  nx, ny, _ = _grid(env, sensor_name)
  return z.reshape(env.num_envs, ny, nx).transpose(1, 2).flatten(1)


def ame2_teacher_mapping(
  env, sensor_name, offset=0.74, clip_z=(-1.5, 1.5),
  void_bias_range=(-1.5, -0.65), custom_noise=None,
):
  z = _height_samples(env, sensor_name, offset, void_bias_range)
  nx, ny, xy = _grid(env, sensor_name)
  z = z.reshape(env.num_envs, ny, nx).transpose(1, 2)
  if clip_z is not None:
    z = z.clamp(*clip_z)
  points = torch.cat((xy[None].expand(env.num_envs, -1, -1, -1), z[..., None]), dim=-1)
  if custom_noise is not None:
    amplitude = torch.as_tensor(custom_noise, device=points.device, dtype=points.dtype)
    points = points + (torch.rand_like(points) * 2 - 1) * amplitude
  return points


def ame2_last_action(env):
  return env.action_manager.prev_action.clamp(-50.0, 50.0)


def ame2_time_left(env):
  return (1.0 - env.episode_length_buf / env.max_episode_length).unsqueeze(1)


def ame2_commands_obs(env, command_name, dist_thr=None):
  command = env.command_manager.get_command(command_name)
  pos_xy, yaw = command[:, :2], command[:, -1]
  if dist_thr is not None:
    distance = pos_xy.norm(dim=1)
    far = distance > dist_thr
    pos_xy = pos_xy * (dist_thr / distance.clamp(min=1e-6)).clamp(max=1).unsqueeze(1)
    yaw = torch.where(far, torch.rand_like(yaw) * (2 * torch.pi) - torch.pi, yaw)
  return torch.cat((pos_xy, yaw.sin()[:, None], yaw.cos()[:, None]), dim=1)


def contact_state_indices(env, sensor_name, sensor_indices=None, mass_threshold=0.1, ignore_bodies=()):
  """Resolve heavy-link channels once using unrandomized masses and sensor names."""
  if sensor_indices is not None:
    return sensor_indices
  key = (sensor_name, mass_threshold, tuple(ignore_bodies))
  if not hasattr(env, "_ame2_contact_indices"):
    env._ame2_contact_indices = {}
  if key not in env._ame2_contact_indices:
    robot = env.scene["robot"]
    masses = env.sim.get_default_field("body_mass")[robot.data.indexing.body_ids].tolist()
    names = [name.split("/")[-1] for name in env.scene[sensor_name].primary_names]
    selected = [names.index(name) for name, mass in zip(robot.body_names, masses)
                if mass > mass_threshold and name not in ignore_bodies]
    env._ame2_contact_indices[key] = selected
  return env._ame2_contact_indices[key]


def ame2_contact_states(env, sensor_name, sensor_indices=None, force_threshold=1.0,
                        mass_threshold=0.1, ignore_bodies=()):
  """Heavy-link contact states; the critic reflection uses this same ordering."""
  ids = contact_state_indices(env, sensor_name, sensor_indices, mass_threshold, ignore_bodies)
  forces = env.scene[sensor_name].data.force_history[:, ids]
  return (forces.norm(dim=-1).amax(dim=-1) > force_threshold).float()
